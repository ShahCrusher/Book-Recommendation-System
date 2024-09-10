from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

#load trained model
popular_df = pickle.load(open('book_model.pkl','rb'))
pt = pickle.load(open('pt.pkl','rb'))
books = pickle.load(open('books.pkl','rb'))
similarity_score = pickle.load(open('similarity_score.pkl','rb'))

book_list = pickle.load(open('book_dict.pkl','rb'))


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',
                           book_name = list(popular_df['Book-Title'].values),
                            author_name = list(popular_df['Book-Author'].values),
                            image = list(popular_df['Image-URL-M'].values),
                            votes = list(popular_df['num_ratings'].values),
                            rating = list(round(popular_df['avg_rating'],1).values)
                           )


@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html',data_keys=book_list.values())

@app.route('/recommend_books',methods=['post'])
def recommend():
    user_input = request.form.get('user_input')
    index = np.where(pt.index==user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1],reverse=True)[1:5]

    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        data.append(item)
        print(data)

    return render_template('recommend.html',data=data,data_keys=book_list.values())
    

    
if __name__ == '__main__':
    app.run(debug=True)
