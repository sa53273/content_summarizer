import tkinter as tk
import nltk
from newspaper import Article
import emoji
from transformers import pipeline

nltk.data.path.append('/Users/suprita/nltk_data/tokenizers/punkt')
emojis = {
            'POSITIVE': '😊',
            'NEGATIVE': '😞',
            'NEUTRAL': '😐'
}
# gui
root = tk.Tk()
root.title('Content Summarizer')
root.geometry('1200x600')

ulabel = tk.Label(root, text='URL')
ulabel.pack()
utext = tk.Text(root, height=1, width=140)
utext.config(bg='#BF522C')
utext.pack()

tlabel = tk.Label(root, text='Title')
tlabel.pack()
title = tk.Text(root, height=1, width=140)
title.config(state='disabled', bg='#BF522C')
title.pack()

alabel = tk.Label(root, text='Author')
author = tk.Text(root, height=1, width=140)
author.config(state='disabled', bg='#BF522C')

slabel = tk.Label(root, text='Summary')
slabel.pack()
summary = tk.Text(root, height=20, width=140)
summary.config(state='disabled', bg='#BF522C')
summary.pack()

selabel = tk.Label(root)
selabel.pack()
sentiment = tk.Label(root, height=1, width=140, font=('Arial', 50))
sentiment.pack()


def summarize():
    print('summarizing...')
    # summarization and sentiment analysis
    url = utext.get('1.0', 'end').strip()
    # parse URL and NLP
    article = Article(url)
    article.download()
    article.parse()
    article.nlp()
    # sentiment analysis w huggingface
    sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    polarity = sentiment_analyzer(article.summary)[0]

    title.config(state='normal')
    author.config(state='normal')
    summary.config(state='normal')
    
    title.delete('1.0', 'end')
    title.insert('1.0', article.title)
    author.delete('1.0', 'end')
    author.insert('1.0', article.authors)
    summary.delete('1.0', 'end')
    summary.insert('1.0', article.summary)
    selabel.config(text=polarity['label'])
    sentiment.config(text=emojis.get(polarity['label']))

    title.config(state='disabled')
    author.config(state='disabled')
    summary.config(state='disabled')

    
    face = emoji.emojize(emojis.get(polarity['label']))
    print(face)

enter = tk.Button(root, text='Summarize', bg='#BF522C', 
                  command=summarize, relief=tk.FLAT, borderwidth=0)
enter.pack()
root.mainloop()