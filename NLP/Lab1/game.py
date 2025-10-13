import tkinter as tk
from tkinter import messagebox
from logic import get_random_word, get_score

class WordGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Word Association Game")
        self.root.geometry("500x400")
        self.root.config(bg="#f8f9fa")

        self.max_rounds = 10
        self.current_round = 1
        self.score_total = 0

        self.create_game_screen()

    def create_game_screen(self):
        self.user_word = tk.StringVar()
        self.word, self.pos, self.definition = get_random_word()

        tk.Label(self.root, text="Word Association Game", font=("Arial", 16, "bold"), bg="#f8f9fa").pack(pady=10)
        self.round_label = tk.Label(self.root, text=f"Round {self.current_round} of {self.max_rounds}", bg="#f8f9fa", font=("Arial", 12))
        self.round_label.pack(pady=5)

        self.word_label = tk.Label(self.root, text=f"Your word: {self.word}", font=("Arial", 14, "bold"), bg="#f8f9fa")
        self.word_label.pack(pady=5)

        self.definition_label = tk.Label(self.root, text=f"Definition: {self.definition}", bg="#f8f9fa", wraplength=450, justify="left")
        self.definition_label.pack(pady=5)

        tk.Label(self.root, text="Type a related word:", bg="#f8f9fa").pack()
        entry = tk.Entry(self.root, textvariable=self.user_word, width=25)
        entry.pack(pady=10)
        entry.focus()

        tk.Button(self.root, text="Submit", command=self.check_word, bg="#28a745", fg="white", relief="flat", padx=10, pady=5).pack(pady=5)

        self.score_label = tk.Label(self.root, text="Score: 0", bg="#f8f9fa", font=("Arial", 10, "italic"))
        self.score_label.pack(pady=5)

    def check_word(self):
        user_input = self.user_word.get().strip().lower()
        if not user_input:
            messagebox.showwarning("Warning", "Please enter a word!")
            return

        score, feedback = get_score(self.word, user_input, self.pos)
        self.score_total += score

        messagebox.showinfo("Result", f"Round {self.current_round} of {self.max_rounds}\n"
                                      f"Similarity Score: {score}\n\n{feedback}")

        if self.current_round < self.max_rounds:
            self.current_round += 1
            self.user_word.set("")
            self.round_label.config(text=f"Round {self.current_round} of {self.max_rounds}")
            avg = round(self.score_total / (self.current_round - 1), 2)
            self.score_label.config(text=f"Total Score: {round(self.score_total, 2)} (Avg: {avg})")
        else:
            avg_final = round(self.score_total / self.max_rounds, 2)
            messagebox.showinfo("Game Over", f"Game Over!\n\nYour total score: {round(self.score_total, 2)}\n"
                                             f"Average similarity: {avg_final}")
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = WordGame(root)
    root.mainloop()
