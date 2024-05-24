import argparse
from faq_core import FAQ
from setence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained", default="", help="Pretrained model from the huggingface")
parser.add_argument("--questions", default="data/diacritics/FAQv5_questions.csv", help="Question data file")
parser.add_argument("--trust_remote_code", default=False, action="store_true", help="Allow remote code execution during model initialization.")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    if args.pretrained:
        if args.trust_remote_code:
            model = SentenceTransformer(args.pretrained,trust_remote_code=True)
        else:
            model = SentenceTransformer(args.pretrained)
    else:
        print("Local models are not supported now!")
        exit(0)

    faq = FAQ(model, args.questions)
    faq.total_confusion()

