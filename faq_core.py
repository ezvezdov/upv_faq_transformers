import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class FAQ:
    def __init__(
            self, 
            model, 
            questions_path, 
            answers_path=None
        ):
        self.model = model
        self.answers = None

        if questions_path.split(".")[1] == "xlsx":
            
            self.questions = pd.read_excel(questions_path)
        elif questions_path.split(".")[1] == "csv":
            self.questions = pd.read_csv(questions_path, sep="\t")
        else:
            raise "Unsupported data file"
        
        if answers_path and questions_path.split(".")[1] == "xlsx":
            self.answers = pd.read_excel(answers_path)
        elif answers_path and questions_path.split(".")[1] == "csv":
            self.answers = pd.read_csv(answers_path, sep="\t")
        elif answers_path:
            raise "Unsupported data file"

        # Create embedding database - matrix of embedding vectors for each question
        self.db = np.array([self.sentence_embedding(q) for q in self.questions["question"]])

        # Mean embedding database - holds averages of question embeddings for each class
        self.mean_db = np.zeros([self.questions["class"].nunique(), self.db.shape[1]])
        for i, cls in enumerate(self.questions["class"].unique()):
            imin = self.questions[self.questions["class"] == cls].index.min()
            imax = self.questions[self.questions["class"] == cls].index.max()
            self.mean_db[i, :] = self.db[imin:imax+1, :].mean(axis=0)

        # Answer database - matrix of embeddings for each unique answer
        if self.answers is not None:
            self.ans_db = np.array([self.sentence_embedding(a) for a in self.answers['answer']])

    def sentence_embedding(self, s):
        return self.model.encode(s, normalize_embeddings=True, show_progress_bar=False)

    def total_confusion(self):
        # Shows a heatmap of cosine similarities of all question pairs
        # Regions within the same class are enclosed in red squares
        # Click a pixel to print out aditional info about the matching question pair
        cm = self.db @ self.db.T
        am = np.argmax(cm, axis=1)
        for i in range(am.shape[0]):
            if am[i] != i:
                print("Ambiguous match:")
                print(self.questions["question"][i], i, self.questions["class"][i])
                print(self.questions["question"][am[i]], am[i], self.questions["class"][am[i]])
                print()

        def onclick(event):
            if event.xdata is None or event.ydata is None:
                return
            x = int(np.round(event.xdata))
            y = int(np.round(event.ydata))
            print(f"Pixel coords: {x}, {y}")
            print(f"Q1: {self.questions['question'][x]}, Class {self.questions['class'][x]}")
            print(f"Q2: {self.questions['question'][y]}, Class {self.questions['class'][y]}")
            print(f'Similarity: {cm[x, y]}')
            print()

        fig = plt.figure(figsize=(10, 7))
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.matshow(cm, 0)
        for cls in self.questions["class"].unique():
            ul = self.questions[self.questions["class"] == cls].index.min() - 0.5
            edge = self.questions[self.questions["class"] == cls].shape[0]
            plt.gca().add_patch(Rectangle((ul, ul), edge, edge, linewidth=1, edgecolor='r', facecolor='none'))
        plt.title("Confusion matrix for all question matches")
        plt.show()

    def mean_match_test(self, verb=False, show_cm=False, show_time=2.0):
        # Determines question class by comparing it with mean database and computes classification accuracy
        cm = self.db @ self.mean_db.T
        am = np.argmax(cm, axis=1)
        preds = am
        gts = self.questions["class"].to_numpy(dtype=int)
        hits = preds == gts

        acc = hits.mean()
        #print(f"Mean match accuracy: {acc}")

        if not hits.all() and verb:
            print("\nIncorrect matches:")
            for i, b in enumerate(hits):
                if not b:
                    print(f"{self.questions['question'][i]} : Class {am[i]}")

        if show_cm:
            cm = confusion_matrix(gts, preds)
            fig = plt.figure(figsize=(10, 7))
            sn.heatmap(cm, annot=True)
            plt.title("Mean matching confusion matrix")
            plt.xlabel("Prediction")
            plt.ylabel("True class")
            plt.draw()
            plt.pause(show_time)
            return acc, fig
        return acc, None
    
    def cross_match_test(self, verb=False, show_cm=False, show_time=2.0):
        # Computes cosine similarities of all question pairs
        # A question is succesfully matched, if its second highest similarity is with a question of the same class
        # Computes accuracy as the ratio of succesfull matches
        cm = self.db @ self.db.T
        am = np.argsort(cm, axis=1)[:, -2]
        cls_ids = self.questions["class"].to_numpy(dtype=int)
        hits = cls_ids == cls_ids[am]

        acc = hits.mean()
        #print(f"Question cross-match accuracy: {acc}")

        if not hits.all() and verb:
            print("\nIncorrect matches:")
            for i, b in enumerate(hits):
                if not b:
                    print(f"{self.questions['question'][i]} : {self.questions['question'][am[i]]}")

        if show_cm:
            cm = confusion_matrix(cls_ids, cls_ids[am])
            fig = plt.figure(figsize=(10, 7))
            sn.heatmap(cm, annot=True)
            plt.title("Question cross-matching confusion matrix")
            plt.xlabel("Prediction")
            plt.ylabel("True class")
            plt.draw()
            plt.pause(show_time)
            return acc, fig
        return acc, None

    def ans_test(self, verb=False, show_cm=False, show_time=2.0):
        # Classifies questions by directly comparing them with embedded answers and computes accuracy
        if self.answers is None:
            warnings.warn("Answers are not available")
            return None
        cm = self.db @ self.ans_db.T
        am = np.argmax(cm, axis=1)
        preds = self.answers["class"].to_numpy(dtype=int)[am]
        gts = self.questions["class"].to_numpy(dtype=int)
        hits = preds == gts

        acc = hits.mean()
        #print(f"Answer match accuracy: {acc}")

        if not hits.all() and verb:
            print("\nIncorrect matches:")
            for i, b in enumerate(hits):
                if not b:
                    print(f"{self.questions['question'][i]} : {self.answers['answer'][am[i]]}")

        if show_cm:
            cm = confusion_matrix(gts, preds)
            fig = plt.figure(figsize=(10, 7))
            sn.heatmap(cm, annot=True)
            plt.title("Answer matching confusion matrix")
            plt.xlabel("Prediction")
            plt.ylabel("True class")
            plt.draw()
            plt.pause(show_time)
            return acc, fig
        return acc, None

    def identify(self, question):
        v = self.sentence_embedding(question)
        sims = self.db @ v[:, np.newaxis]
        return np.argmax(sims)
    
    def identify_direct_answer(self, question):
        v = self.sentence_embedding(question)
        a_sims = self.ans_db @ v[:, np.newaxis]
        return np.argmax(a_sims)
    
    def match(self, question):
        matched_q = self.questions['question'][self.identify(question)]
        #print(f"Matched question: {matched_q}")
        return matched_q

    def answer(self, question):
        if self.answers is None:
            warnings.warn("Answers are not available")
            return None
        ans = self.answers['answer'][self.questions['class'][self.identify(question)]]
        #print(f"Answer: {ans}")
        return ans
    
    def direct_answer(self, question):
        if self.answers is None:
            warnings.warn("Answers are not available")
            return None
        ans = self.answers['answer'][self.identify_direct_answer(question)]
        #print(f"Answer: {ans}")
        return ans
