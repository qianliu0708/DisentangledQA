from typing import List


class Argument:
    pretrained_model_path: str = './pretrained_model/6_STAR_ORA-P/weights.th'
    model_path: str = './checkpoints/combination/1.pth'
    model_class: str = "ReasoningPlain"

    train_path: str = "./data/train_gdsent.json"  # data/transformer_qa_ORA-P_train_no_placeholders.json
    dev_path: str = "./data/dev_gdsent.json"
    test_path: str = "./data/test.json"
    corpus_path: str = "./data/strategyqa/strategyqa_train_paragraphs.json"

    train_gdsent_path: str = "./data/train_gdsent.json"
    dev_gdsent_path: str = "./data/dev_gdsent.json"

    train_dataset: str = "reasoning_dataset"
    dev_dataset: str = "reasoning_dataset"
    test_dataset: str = "reasoning_dataset"
    fields: List = ["question", "evidence"]
    max_length: int = 512

    boolq_path: str = './data/boolq/train.jsonl'
    twentyquestion_path: str = './data/twentyquestions/v1.0.twentyquestions.tar'
    squad_train_path: str = './data/squad/train-v2.0.json'
    squad_dev_path: str = './data/squad/dev-v2.0.json'

    cuda: bool = False
    num_workers: int = 0
    load_pretrained: bool = True

    pretrain_epoch_num_boolq: int = 20
    pretrain_epoch_num_20q: int = 20
    pretrain_batch_size: int = 8
    pretrain_learning_rate: float = 1e-5

    epoch_num: int = 20
    batch_size: int = 1
    learning_rate: float = 1e-5
    tuning_rate: float = 1e-5

    warmup_rate: float = 0.1
    weight_decay: float = 0.01

    ir_avgcls_dev_path: str = './sqa_ir/5_dev_candidate_path_avg.json'
    ir_avgcls_test_path: str = './sqa_ir/5_test_candidate_path_avg.json'
    paraid_content_path: str = './sqa_ir/paraid_content.pk'
    prediction_path: str = './predictions/5_avg_prediction.json'

    sentchain_train: str = './sqa_ir/7_train_sentchain.json'
    sentchain_dev: str = './sqa_ir/7_dev_sentchain.json'
    sentchain_test: str = ''

    reason_train: str = './data/reason/train_sents.pk'
    reason_dev: str = './data/reason/dev_sents.pk'
    reason_test: str = './data/reason/test_sents.pk'