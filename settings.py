import os


DATA_NAME = os.environ.get("DATA_NAME", "USPTO")
EXP_NAME = os.environ.get("EXP_NAME", "")

SCALE = 2
SAMPLE_SIZE = 64 // SCALE
NUM_GPU = int(os.environ.get("NUM_GPUS_PER_NODE", 1))

TRAIN_BATCH_SIZE = 4096 * NUM_GPU
VAL_BATCH_SIZE = 4096 * NUM_GPU
TEST_BATCH_SIZE = (512 * NUM_GPU * SCALE)

NUM_NODES = int(os.environ.get("NUM_NODES", 1))
ACCUMULATION_COUNT = int(os.environ.get("ACCUMULATION_COUNT", 1))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 16))

# MODEL_NAME = ""
MODEL_NAME = "model.1440000_47.pt"


class Args:
    # train #
    model_name = MODEL_NAME
    exp_name = EXP_NAME
    train_path = f"data/{DATA_NAME}/train.txt" 
    val_path = f"data/{DATA_NAME}/val.txt"
    # test_path = f"data/{DATA_NAME}/test.txt"
    test_path = f"data/{DATA_NAME}/beam.txt"
    processed_path = f"data/{DATA_NAME}/processed" 
    model_path = f"{os.getcwd()}/checkpoints/{DATA_NAME}/{EXP_NAME}/"
    result_path = f"results/{DATA_NAME}/{EXP_NAME}"
    data_name = f"{DATA_NAME}"
    log_file = f"FlowER_{DATA_NAME}_{EXP_NAME}"
    load_from = ""
    # load_from = f"{model_path}{MODEL_NAME}"
    # resume = True

    backend = "nccl"
    num_workers = NUM_WORKERS
    emb_dim = 128
    enc_num_layers = 12
    enc_heads = 32
    enc_filter_size = 2048
    dropout = 0.0
    attn_dropout = 0.0
    rel_pos = "emb_only"
    shared_attention_layer = 0
    sigma = 0.15
    train_batch_size = (TRAIN_BATCH_SIZE / ACCUMULATION_COUNT / NUM_GPU / NUM_NODES)
    val_batch_size = (VAL_BATCH_SIZE / ACCUMULATION_COUNT / NUM_GPU / NUM_NODES)
    test_batch_size = TEST_BATCH_SIZE
    batch_type = "tokens_sum"
    lr = 0.0001
    beta1 = 0.9
    beta2 = 0.998
    eps = 1e-9
    weight_decay = 1e-2
    warmup_steps = 30000
    clip_norm = 200


    epoch = 100
    max_steps = 1500000
    accumulation_count = ACCUMULATION_COUNT
    save_iter = 30000
    log_iter = 100
    eval_iter = 30000

    sample_size = SAMPLE_SIZE
    sym_break_noise_stdv = 0.02

    rbf_low = 0
    rbf_high = 8
    rbf_gap = 0.1


    # validation #
    # do_validate = True
    # steps2validate =  ["1050000", "1320000", "1500000", "930000", "1020000"]
    
    # inference # 
    do_validate = False

    # beam-search #
    beam_size = 5
    nbest = 3
    max_depth = 15
    chunk_size = 50