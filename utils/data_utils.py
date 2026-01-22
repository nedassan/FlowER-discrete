import torch
from torch.utils.data import Dataset
from rdkit import Chem
import numpy as np
import sys
import os
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 
from multiprocessing import Pool, cpu_count
from utils.arrow_pushing import get_arrow_pushing

np.set_printoptions(threshold=sys.maxsize, linewidth=500)
torch.set_printoptions(profile="full")

ELEM_LIST = ['PAD', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', \
             'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', \
             'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Sr', 'Y', 'Zr', 'Mo', 'Tc', 'Ru', \
             'Rh', 'Pd', 'Ag', 'In', 'Sn', 'Sb', 'Te', 'I', 'Cs', 'Ba', 'La', 'Ce', 'Eu', \
              'Yb', 'Ta', 'W', 'Os', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'V', 'Sm']

MATRIX_PAD = -30
ARROW_PAD = -1

bt_to_electron = {Chem.rdchem.BondType.SINGLE: 2, 
                 Chem.rdchem.BondType.DOUBLE: 4,
                 Chem.rdchem.BondType.TRIPLE: 6,
                 Chem.rdchem.BondType.AROMATIC: 3}

tbl = Chem.GetPeriodicTable()

def bond_features(bond):
    bt = bond.GetBondType()
    return bt_to_electron[bt]

def count_lone_pairs(a):
    v=tbl.GetNOuterElecs(a.GetAtomicNum())
    c=a.GetFormalCharge()
    b=sum([bond.GetBondTypeAsDouble() for bond in a.GetBonds()])
    h=a.GetTotalNumHs()
    return v-c-b-h

ps = Chem.SmilesParserParams()
ps.removeHs = False
ps.sanitize = True

def get_BE_matrix(r):
    rmol = Chem.MolFromSmiles(r, ps)
    Chem.Kekulize(rmol)
    max_natoms = len(rmol.GetAtoms())
    f = np.zeros((max_natoms,max_natoms))
    
    for atom in rmol.GetAtoms():
        lone_pair = count_lone_pairs(atom)
        f[atom.GetIntProp('molAtomMapNumber') - 1, atom.GetIntProp('molAtomMapNumber') - 1] = lone_pair

    for bond in rmol.GetBonds():
        a1 = bond.GetBeginAtom().GetIntProp('molAtomMapNumber') - 1
        a2 = bond.GetEndAtom().GetIntProp('molAtomMapNumber') - 1
        f[(a1,a2)] = f[(a2,a1)] = bond_features(bond)/2 

    return f

electron_to_bo = {val:key for key, val in bt_to_electron.items()}

def get_formal_charge(a, electron):
    v=tbl.GetNOuterElecs(a.GetAtomicNum())
    b=sum([bond.GetBondTypeAsDouble() for bond in a.GetBonds()])
    h=a.GetTotalNumHs()
    f =v -  electron  - b - h
    return f

def mol_prop_compute(matrix):
    n = matrix.shape[0]
    Mplus = matrix + matrix.T
    iu, ju = np.triu_indices(n, k=1)
    vals = Mplus[iu, ju]
    mask = vals != 0
    bond_dict = {
        (i + 1, j + 1): int(val)
        for i, j, val in zip(iu[mask], ju[mask], vals[mask])
    }
    diag = matrix.diagonal()
    atom_dict = {
            (i + 1, i + 1): int(diag_val)
        for i, diag_val in enumerate(diag)
    }
    return atom_dict, bond_dict

def BEmatrix_to_mol(rmol, matrix, idxfunc=lambda x:x.GetIdx()):
    atom_dict, bond_dict = mol_prop_compute(matrix)
                
    new_mol = Chem.RWMol(rmol)
    new_mol.UpdatePropertyCache(strict=False)
    
    amap = {}
    for atom in new_mol.GetAtoms():
        amap[atom.GetIntProp('molAtomMapNumber') - 1] = atom.GetIdx()

    for bond in rmol.GetBonds():
        a1 = idxfunc(bond.GetBeginAtom())
        a2 = idxfunc(bond.GetEndAtom())
        new_mol.RemoveBond(a1, a2)
        
    for (a1, a2), electron in bond_dict.items():
        new_mol.AddBond(amap[a1-1], amap[a2-1], electron_to_bo[electron])
        
    for (a1, a1), electron in atom_dict.items():
        a =  new_mol.GetAtomWithIdx(amap[a1-1])
        fc = get_formal_charge(a, electron)
        a.SetFormalCharge(int(fc))
    return new_mol

atom2idx_dict = {elem:i for i, elem in enumerate(ELEM_LIST)}
def smi2vocabid(smi):
    mol = Chem.MolFromSmiles(smi, ps)
    smi_vocab_id_list = np.zeros(len(mol.GetAtoms()))
    for atom in mol.GetAtoms():
        idx = atom2idx_dict[atom.GetSymbol()]
        smi_vocab_id_list[atom.GetIntProp('molAtomMapNumber') - 1] = idx
    return smi_vocab_id_list, len(smi_vocab_id_list)

def process_arrows(arrow_list):
    flat_arrows = []
    for arrow in arrow_list:
        sources = arrow['src'] if isinstance(arrow['src'], list) else [arrow['src']]
        sinks = arrow['sink'] if isinstance(arrow['sink'], list) else [arrow['sink']]
        
        count = arrow['electrons']
        num_sub_arrows = max(len(sources), len(sinks))
        if count % num_sub_arrows != 0:
            e_per_arrow = count / num_sub_arrows
        else:
            e_per_arrow = count // num_sub_arrows

        for s in sources:
            for k in sinks:
                s_u, s_v = sorted(s)
                k_u, k_v = sorted(k)
                flat_arrows.append([s_u, s_v, k_u, k_v, e_per_arrow])
                
    if len(flat_arrows) == 0:
        return np.zeros((0, 5), dtype=np.float32)
        
    return np.array(flat_arrows, dtype=np.float32)

def process_smiles(smiles):
    line = smiles.strip()
    if not line or ">>" not in line: return None
    
    src_smi, tgt_smi = line.split('|')[0].split('>>')

    try:
        src_vocab_id_list, src_len = smi2vocabid(src_smi)
        tgt_vocab_id_list, tgt_len = smi2vocabid(tgt_smi)
        assert (src_vocab_id_list == tgt_vocab_id_list).all()
        
        return {
            'src_smi': src_smi,
            'tgt_smi': tgt_smi,
            'src_vocab_id_list': src_vocab_id_list,
            'tgt_vocab_id_list': tgt_vocab_id_list,
            'src_len': src_len,
            'tgt_len': tgt_len
        }
    except Exception as e:
        return None

class ReactionBatch:
    def __init__(self,
                 src_data_indices: torch.Tensor,
                 src_token_ids: torch.Tensor,
                 src_lens: torch.Tensor,
                 src_matrices: torch.Tensor,
                 tgt_matrices: torch.Tensor,
                 matrix_masks: torch.Tensor,
                 src_arrows: torch.Tensor,
                 src_arrow_lens: torch.Tensor,
                 src_smiles_list: list,
                 tgt_smiles_list: list,
                 ):
        self.src_data_indices = src_data_indices
        self.src_token_ids = src_token_ids
        self.src_lens = src_lens
        self.src_matrices = src_matrices
        self.tgt_matrices = tgt_matrices
        self.matrix_masks = matrix_masks
        self.src_arrows = src_arrows
        self.src_arrow_lens = src_arrow_lens
        self.src_smiles_list = src_smiles_list
        self.tgt_smiles_list = tgt_smiles_list

    def to(self, device):
        self.src_data_indices = self.src_data_indices.to(device)
        self.src_token_ids = self.src_token_ids.to(device)
        self.src_lens = self.src_lens.to(device)
        self.src_matrices = self.src_matrices.to(device)
        self.tgt_matrices = self.tgt_matrices.to(device)
        self.matrix_masks = self.matrix_masks.to(device)
        if self.src_arrows is not None:
            self.src_arrows = self.src_arrows.to(device)
            self.src_arrow_lens = self.src_arrow_lens.to(device)

    def pin_memory(self):
        self.src_data_indices = self.src_data_indices.pin_memory()
        self.src_token_ids = self.src_token_ids.pin_memory()
        self.src_lens = self.src_lens.pin_memory()
        self.src_matrices = self.src_matrices.pin_memory()
        self.tgt_matrices = self.tgt_matrices.pin_memory()
        self.matrix_masks = self.matrix_masks.pin_memory()
        if self.src_arrows is not None:
            self.src_arrows = self.src_arrows.pin_memory()
            self.src_arrow_lens = self.src_arrow_lens.pin_memory()
        return self

class ReactionDataset(Dataset):
    def __init__(self, args, smiles_list=None, parallel=True, reactant_only=False, cache_path=None):
        self.args = args
        self.device = args.device
        self.reactant_only = reactant_only
        self.batched = False
        
        if cache_path and os.path.exists(cache_path):
            print(f"Loading pre-processed dataset from {cache_path}...")
            cached_data = torch.load(cache_path)
            
            self.src_smis = cached_data['src_smis']
            self.tgt_smis = cached_data['tgt_smis']
            self.src_token_ids = cached_data['src_token_ids']
            self.tgt_token_ids = cached_data['tgt_token_ids']
            
            self.src_lens = np.array(cached_data['src_lens'])
            self.tgt_lens = np.array(cached_data['tgt_lens'])
            
            self.src_matrices = cached_data['src_matrices']
            self.tgt_matrices = cached_data['tgt_matrices']
            self.src_arrows = cached_data['src_arrows']
            
            self.data_size = len(self.src_smis)
            self.data_indices = np.arange(self.data_size)
            print("Loaded successfully.")
            return

        print("No cache found. Parsing raw SMILES (Slow)...")
        if smiles_list is None:
            raise ValueError("Cache not found and no smiles_list provided!")
            
        self.smiles_list = smiles_list
        self.src_smis = []
        self.tgt_smis = []
        self.src_token_ids = []
        self.tgt_token_ids = []
        self.src_lens = []
        self.tgt_lens = []

        self.src_matrices = [None] * len(smiles_list) 
        self.tgt_matrices = [None] * len(smiles_list)
        self.src_arrows = [None] * len(smiles_list)

        if reactant_only:
            self.parse_reactant_only()
        else:
            if parallel:
                self.parse_data_parallel()
            else:
                self.parse_data()
        
        self.src_lens = np.asarray(self.src_lens)
        self.data_size = len(self.src_smis)
        self.data_indices = np.arange(self.data_size)

    def parse_reactant_only(self):
        for smile in self.smiles_list:
            res = process_smiles(smile + ">>C")
            if res:
                self.src_smis.append(res['src_smi'])
                self.tgt_smis.append(res['tgt_smi'])
                self.src_token_ids.append(res['src_vocab_id_list'])
                self.tgt_token_ids.append(res['tgt_vocab_id_list'])
                self.src_lens.append(res['src_len'])
                self.tgt_lens.append(res['tgt_len'])

    def parse_data(self):
        for smile in self.smiles_list:
            res = process_smiles(smile)
            if res:
                self.src_smis.append(res['src_smi'])
                self.tgt_smis.append(res['tgt_smi'])
                self.src_token_ids.append(res['src_vocab_id_list'])
                self.tgt_token_ids.append(res['tgt_vocab_id_list'])
                self.src_lens.append(res['src_len'])
                self.tgt_lens.append(res['tgt_len'])

    def parse_data_parallel(self):
        with Pool(cpu_count()) as p:
            results = p.map(process_smiles, self.smiles_list)
        
        for res in results:
            if res:
                self.src_smis.append(res['src_smi'])
                self.tgt_smis.append(res['tgt_smi'])
                self.src_token_ids.append(res['src_vocab_id_list'])
                self.tgt_token_ids.append(res['tgt_vocab_id_list'])
                self.src_lens.append(res['src_len'])
                self.tgt_lens.append(res['tgt_len'])

    def __len__(self):
        if hasattr(self, 'batch_starts'):
            return len(self.batch_starts)
        return self.data_size

    def sort(self):
        self.data_indices = np.argsort(self.src_lens)

    def shuffle_in_bucket(self, bucket_size=1000):
        for i in range(0, self.data_size, bucket_size):
            np.random.shuffle(self.data_indices[i:i + bucket_size])

    def batch(self, batch_type, batch_size):
        self.batched = True
        self.batch_starts = []
        self.batch_ends = []
        
        if batch_type == "tokens":
            curr_len = 0
            start = 0
            for i, idx in enumerate(self.data_indices):
                curr_len += self.src_lens[idx]
                if curr_len > batch_size:
                    self.batch_starts.append(start)
                    self.batch_ends.append(i)
                    start = i
                    curr_len = self.src_lens[idx]
            self.batch_starts.append(start)
            self.batch_ends.append(self.data_size)
            
        elif batch_type == "tokens_sum":
            curr_len = 0
            start = 0
            for i, idx in enumerate(self.data_indices):
                l = self.src_lens[idx]
                curr_len += l
                if curr_len > batch_size:
                    self.batch_starts.append(start)
                    self.batch_ends.append(i)
                    start = i
                    curr_len = l
            self.batch_starts.append(start)
            self.batch_ends.append(self.data_size)
            
        else:
            self.batch_starts = np.arange(0, self.data_size, batch_size)
            self.batch_ends = np.arange(batch_size, self.data_size + batch_size, batch_size)
            self.batch_ends[-1] = self.data_size

    def __getitem__(self, idx : int):
        if self.batched:
            if hasattr(self, 'batch_starts'):
                batch_start = self.batch_starts[idx]
                batch_end = self.batch_ends[idx]
                data_indices = self.data_indices[batch_start:batch_end]
            elif hasattr(self, 'batches'):
                data_indices = self.batches[idx]
            else:
                raise ValueError("Dataset is 'batched' but no batching logic found.")
        else:
            data_indices = [idx]
        
        if len(data_indices) == 0:
            data_indices = [0]

        max_len = max(self.src_lens[data_indices])

        src_token_id_batch = []
        src_len_batch = []
        src_matrix_batch = []
        tgt_matrix_batch = []
        src_smiles_batch = []
        tgt_smiles_batch = []
        
        src_arrow_list = []
        src_arrow_len_batch = []

        for data_index in data_indices:
            src_token_id = self.src_token_ids[data_index]
            src_len = self.src_lens[data_index]
            src_token_id = np.pad(src_token_id, (0, max_len - src_len),
                                    mode='constant', constant_values=0)
            src_token_id_batch.append(torch.as_tensor(src_token_id, dtype=torch.long))

            if self.src_matrices[data_index] is not None:
                src_matrix = self.src_matrices[data_index]
            else:
                src_matrix = torch.from_numpy(get_BE_matrix(self.src_smis[data_index])).float()

            pad_r = max_len - src_len
            if pad_r > 0:
                src_matrix = torch.nn.functional.pad(src_matrix, (0, pad_r, 0, pad_r), value=MATRIX_PAD)
            
            src_matrix_batch.append(src_matrix)
            src_len_batch.append(src_len)
            src_smiles_batch.append(self.src_smis[data_index])

            if not self.reactant_only:
                if self.tgt_matrices[data_index] is not None:
                    tgt_matrix = self.tgt_matrices[data_index]
                else:
                    tgt_matrix = torch.from_numpy(get_BE_matrix(self.tgt_smis[data_index])).float()
                
                if pad_r > 0:
                    tgt_matrix = torch.nn.functional.pad(tgt_matrix, (0, pad_r, 0, pad_r), value=MATRIX_PAD)
                
                tgt_matrix_batch.append(tgt_matrix)
                tgt_smiles_batch.append(self.tgt_smis[data_index])
                
                if self.src_arrows[data_index] is not None:
                    arrow_tensor = self.src_arrows[data_index]
                else:
                    m_src = src_matrix[:src_len, :src_len].numpy()
                    m_tgt = tgt_matrix[:src_len, :src_len].numpy()
                    
                    delta = m_tgt - m_src
                    raw_arrows = get_arrow_pushing(delta)
                    arrow_tensor = torch.from_numpy(process_arrows(raw_arrows)).float()
                
                src_arrow_list.append(arrow_tensor)
                src_arrow_len_batch.append(len(arrow_tensor))

        src_data_indices = torch.as_tensor(data_indices, dtype=torch.long)
        src_len_batch = torch.as_tensor(src_len_batch, dtype=torch.long)
        src_token_id_batch = torch.stack(src_token_id_batch)
        src_matrix_batch = torch.stack(src_matrix_batch)
        
        if not self.reactant_only: 
            tgt_matrix_batch = torch.stack(tgt_matrix_batch)
            
            max_arrows = max(src_arrow_len_batch) if len(src_arrow_len_batch) > 0 else 0
            if max_arrows > 0:
                padded_arrows = []
                for arrow_arr in src_arrow_list:
                    pad_len = max_arrows - arrow_arr.shape[0]
                    if pad_len > 0:
                        padded = torch.nn.functional.pad(arrow_arr, (0, 0, 0, pad_len), value=-1) # ARROW_PAD
                        padded_arrows.append(padded)
                    else:
                        padded_arrows.append(arrow_arr)
                src_arrow_batch = torch.stack(padded_arrows)
            else:
                src_arrow_batch = torch.zeros((len(src_arrow_len_batch), 0, 5), dtype=torch.float32)
                
            src_arrow_len_batch = torch.as_tensor(src_arrow_len_batch, dtype=torch.long)
        else: 
            tgt_matrix_batch = src_matrix_batch
            src_arrow_batch = None
            src_arrow_len_batch = None
        
        node_mask = (src_matrix_batch[:, :, 0] != MATRIX_PAD)
        matrix_masks = (node_mask.unsqueeze(1) * node_mask.unsqueeze(2)).long()

        reaction_batch = ReactionBatch(
            src_data_indices=src_data_indices,
            src_token_ids=src_token_id_batch,
            src_lens=src_len_batch,
            src_matrices=src_matrix_batch,
            tgt_matrices=tgt_matrix_batch,
            matrix_masks=matrix_masks,
            src_arrows=src_arrow_batch,
            src_arrow_lens=src_arrow_len_batch,
            src_smiles_list=src_smiles_batch,
            tgt_smiles_list=tgt_smiles_batch
        )
        
        return reaction_batch
