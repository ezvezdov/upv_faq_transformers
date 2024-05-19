from sentence_transformers import SentenceTransformer
from faq_core import FAQ

model = SentenceTransformer("Seznam/simcse-retromae-small-cs")#, trust_remote_code=True)

faq = FAQ(model, "data/diacritics/FAQ50_questions.csv", "data/diacritics/FAQ50_answers.csv")

test_question = "Jak požádat o patent?"
matched = faq.match(test_question)
answer = faq.answer(test_question)
direct_answer = faq.direct_answer(test_question)

print(f"\nQuestion:\n{test_question}")
print(f"\nBest Match:\n{matched}")
print(f"\nBest Match Answer:\n{answer}")
print(f"\nBest Directly Matched Answer:\n{direct_answer}")

acc, cm = faq.cross_match_test()
print(f"\nQuestion Cross-Match Accuracy: {acc}")