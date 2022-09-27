import argparse
from trainer.trainer import Trainer


def main(args):
    print(args)
    t = Trainer(args)
    t.train()
    # t.pretrain()
    t.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StrategyQA')
    parser.add_argument('--pretrained_model_path', type=str, default='./pretrained_model/6_STAR_ORA-P/weights.th')
    parser.add_argument('--model_path', type=str, default='./checkpoints/combination/1.pth')
    parser.add_argument('--trained_model_path', type=str, default='./checkpoints/combination/1.pth')
    parser.add_argument('--model_class', type=str, default="ReasoningPlain")

    parser.add_argument('--train_path', type=str, default="./data/train_gdsent.json")  # data/transformer_qa_ORA-P_train_no_placeholders.json
    parser.add_argument('--dev_path', type=str, default="./data/dev_gdsent.json")
    parser.add_argument('--test_path', type=str, default="./data/test.json")
    parser.add_argument('--corpus_path', type=str, default="./data/strategyqa/strategyqa_train_paragraphs.json")

    parser.add_argument('--train_gdsent_path', type=str, default="./data/train_gdsent.json")
    parser.add_argument('--dev_gdsent_path', type=str, default="./data/dev_gdsent.json")

    parser.add_argument('--train_dataset', type=str, default="reasoning_dataset")
    parser.add_argument('--dev_dataset', type=str, default="reasoning_dataset")
    parser.add_argument('--test_dataset', type=str, default="reasoning_dataset")
    parser.add_argument('--fields', type=list, default= ["question", "evidence"])
    parser.add_argument('--max_length', type=int, default=512)

    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--load_pretrained', type=bool, default=True)
    parser.add_argument('--eval_only', type=bool, default=False)

    parser.add_argument('--epoch_num', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--tuning_rate', type=float, default=1e-5)
    parser.add_argument('--weightofoperator', type=float, default=1.)
    parser.add_argument('--op_classification', type=str, default='./classification/4.json')
    parser.add_argument('--gradient_accumulate_step', type=int, default=1)

    parser.add_argument('--warmup_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--ir_avgcls_dev_path', type=str, default='./sqa_ir/5_dev_candidate_path_avg.json')
    parser.add_argument('--ir_avgcls_test_path', type=str, default='./sqa_ir/5_test_candidate_path_avg.json')
    parser.add_argument('--paraid_content_path', type=str, default='./sqa_ir/paraid_content.pk')
    parser.add_argument('--prediction_path', type=str, default='./predictions/5_avg_prediction.json')

    parser.add_argument('--sentchain_train', type=str, default='./sqa_ir/7_train_sentchain.json')
    parser.add_argument('--sentchain_dev', type=str, default='./sqa_ir/7_dev_sentchain.json')
    parser.add_argument('--sentchain_test', type=str, default='')

    parser.add_argument('--reason_train', type=str, default='./data/reason/train_sents.pk')
    parser.add_argument('--reason_dev', type=str, default='./data/reason/dev_sents.pk')
    parser.add_argument('--reason_test', type=str, default='./data/reason/test_sents.pk')

    args = parser.parse_args()
    main(args)