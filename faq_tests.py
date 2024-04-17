from sentence_transformers import SentenceTransformer
import os
import argparse
from faq_core import FAQ


parser = argparse.ArgumentParser()
parser.add_argument("--pretrained", default="", help="Pretrained model from the huggingface")
parser.add_argument("--questions", default="data/diacritics/FAQv5_questions.csv", help="Question data file")
parser.add_argument("--answers", default="data/diacritics/FAQv5_answers.csv", help="Answer data file")
parser.add_argument("--verb", default=False, action="store_true", help="Print incorrect matches")
parser.add_argument("--cm", default=False, action="store_true", help="Create and show a confusion matrix")
parser.add_argument("--cmtime", default=0.0, type=float, help="Confusion matrix display duration")
parser.add_argument("--save_dir", default="output/", help="Save results to the provided directory")
parser.add_argument("--filename", default="Accuracies.log", help="Filename of the saved results")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    
    if args.pretrained:
        model = SentenceTransformer(args.pretrained)
    else:
        print("Local models are not supported now!")
        exit(0)

    faq = FAQ(model, args.questions, args.answers)
    q_acc, q_cm = faq.cross_match_test(verb=args.verb, show_cm=args.cm, show_time=args.cmtime)
    a_acc, a_cm = faq.ans_test(verb=args.verb, show_cm=args.cm, show_time=args.cmtime) 

    
    print(f"Question cross-match accuracy: {round(q_acc,6)}")
    print(f"Answer match accuracy: {round(a_acc,6)}")

    if args.save_dir:
        save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, args.filename), "w") as af:
            af.writelines([f"Question matching accuracy: {round(q_acc,6)} \nAnswer matching accuracy: {round(a_acc,6)}\n"])
        if args.cm:
            q_cm.savefig(os.path.join(save_dir, "Question_CM.png"))
            a_cm.savefig(os.path.join(save_dir, "Answer_CM.png"))
