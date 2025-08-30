import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from typing import List, Generator

from utils.data_utils import get_BE_matrix
from utils.arrow_pushing import get_arrow_pushing

# EXAMPLE LINE FROM DATA:
# [C:19]([O:20][H:36])([C:21]([H:37])([H:38])[H:39])([H:34])[H:35].[C:1]([C:2]([O:3][C:4]([C:5]([H:26])([H:27])[H:28])([H:24])[H:25])=[O:6])([Br:7])([H:22])[H:23].[C:8]([N:9]1[C:10](=[O:11])[N:12]([H:32])[C:13](=[O:14])[C:15]1=[O:16])([H:29])([H:30])[H:31].[K+:17].[O-:18][H:33]
# >>
# [Br-:7].[C:19]([O:20][H:36])([C:21]([H:37])([H:38])[H:39])([H:34])[H:35].[C:1]([C:2]([O:3][C:4]([C:5]([H:26])([H:27])[H:28])([H:24])[H:25])=[O:6])([N+:12]1([H:32])[C:10](=[O:11])[N:9]([C:8]([H:29])([H:30])[H:31])[C:15](=[O:16])[C:13]1=[O:14])([H:22])[H:23].[K+:17].[O-:18][H:33]
# |1

TEST_NAME = "test.txt"
TRAIN_NAME = "train.txt"
VAL_NAME = "val.txt"

def file_smi_generator(data_dir: str) -> Generator[tuple[str, str], None, None]:
    files = [
        os.path.join(data_dir, TEST_NAME),
        os.path.join(data_dir, TRAIN_NAME),
        os.path.join(data_dir, VAL_NAME)
    ]

    for file in files:

        if not os.path.exists(file):
            print(f"Warning: File not found at {file}. Skipping.")
            continue

        with open(file, 'r') as f:
            for line in f:
                try:
                    reactant_smi, product_smi = line.strip().split('|')[0].split('>>')
                    yield reactant_smi, product_smi
                except ValueError:
                    continue

def count_bond_types(data_dir: str) -> List[int]:
    bond_types = set()

    for reactant_smi, product_smi in file_smi_generator(data_dir):
        try:
            reactant_be_matrix = get_BE_matrix(reactant_smi)
            product_be_matrix = get_BE_matrix(product_smi)
            bond_types.update(np.unique(reactant_be_matrix))
            bond_types.update(np.unique(product_be_matrix))
        except Exception as e:
            pass 
    
    return sorted(list(bond_types))
# RESULT = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 14.0]

def count_deltas(data_dir: str) -> List[int]:
    bond_deltas = set()

    for reactant_smi, product_smi in file_smi_generator(data_dir):
        try:
            reactant_be_matrix = get_BE_matrix(reactant_smi)
            product_be_matrix = get_BE_matrix(product_smi)
            delta_matrix = product_be_matrix - reactant_be_matrix
            bond_deltas.update(np.unique(delta_matrix))
        except Exception as e:
            pass 

    return sorted(list(bond_deltas))
# RESULT = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]

def test_arrow_pushing(data_dir: str, idx: int = 0) -> Generator[dict, None, None]:
    iter_idx = 0 
    for reactant_smi, product_smi in file_smi_generator(data_dir):
        if idx == iter_idx:
            try:
                reactant_be_matrix = get_BE_matrix(reactant_smi)
                product_be_matrix = get_BE_matrix(product_smi)
                delta_matrix = product_be_matrix - reactant_be_matrix
                delta_matrix[np.triu_indices(delta_matrix.shape[0], k=1)] *= 2
                moves = get_arrow_pushing(delta_matrix)
                yield {
                    'reactant': reactant_smi,
                    'product': product_smi,
                    'moves': moves
                }
            except Exception as e:
                raise ValueError("Malformed entry")
        iter_idx += 1

from rdkit.Chem import Draw

def draw_mapped_reaction(reactant_smi, product_smi, moves=None, filename='reaction.png'):
    reactant_mol = Chem.MolFromSmiles(reactant_smi)
    product_mol = Chem.MolFromSmiles(product_smi)
    
    opts = Draw.MolDrawOptions()
    opts.addAtomIndices = True
    opts.includeAtomicNumbers = False

    img = Draw.MolsToGridImage(
        [reactant_mol, product_mol],
        molsPerRow=2,
        subImgSize=(300, 300),
        legends=["Reactant", "Product"],
        useSVG=False
    )
    img.save(filename)
    print(f"Saved reaction image to {filename}")

    if moves:
        print("Arrow pushing moves:")
        for m in moves:
            shifted_src = []
            for idx, val in enumerate(m['src']):
                shifted_src.append(m['src'][idx] + 1)
            shifted_sink = []
            for idx, val in enumerate(m['sink']):
                shifted_sink.append(m['sink'][idx] + 1)
            
            m['src'] = tuple(shifted_src)
            m['sink'] = tuple(shifted_sink)

            print(m)

if __name__ == "__main__":
    # print(count_bond_types("flower_new_dataset"))
    # print(count_deltas("flower_new_dataset"))
    arrow_pushing_dict = next(test_arrow_pushing("flower_new_dataset", 12))
    draw_mapped_reaction(
        arrow_pushing_dict["reactant"], 
        arrow_pushing_dict["product"], 
        moves=arrow_pushing_dict["moves"]
    )
