from flask import Flask, render_template,request
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
import requests

app = Flask(__name__)

data=pd.read_csv("moives_processed.csv")
#popularity based
p_df = data[[ 'original_title','vote_average','vote_count']]

# calculating all the components based IMDB formula
v= p_df['vote_count']
R= p_df['vote_average']
C= p_df['vote_average'].mean()
m= p_df['vote_count'].quantile(0.70)

p_df['Weighted_Average']= ((R*v)+ (C*m))/(v+m)
popular_movies = p_df.sort_values(by='Weighted_Average',ascending=False)


##content based 
vectorizer = TfidfVectorizer(max_features=1000)
movie_vectors = vectorizer.fit_transform(data['content'].values)
similarity = cosine_similarity(movie_vectors)
similarity_df = pd.DataFrame(similarity)
names =[]



def recommend(movie):
     # Converting uppercase into lower
    movie = movie.lower()
    try:
       # Correcting user input spell (close match from our movie list)
       movie = get_close_matches(movie, data['title'].values, n=3, cutoff=0.6)[0]
       # find movie index from dataset
       movies_index = data[data['title'] == movie].index[0]
       # finding cosine similarities of movie
       distances = similarity[movies_index]
            
       # sorting cosine similarities
       movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[0:9]
            
       names.clear()
       for i in movies_list:

          results =requests.get("http://www.omdbapi.com/?apikey=629b6ae&t="+(data.iloc[i[0]].title)).json()
          names.append({'poster':results['Poster'],
                           'title':results['Title'],
                           'overview':results['Plot'],
                           'rating':results['imdbRating'],
                           'votes':results['imdbVotes'],
                           'genre':results['Genre'],
                           'date':results['Released'],
                           'actors':results['Actors'],
                           'director':results['Director']})
    except:
       names.clear()

      

    
   
@app.route("/",methods = ['POST', 'GET'])
def index():
   if request.method == 'POST':
      moviename = request.form['moviename']
      recommend(moviename)
      if names:
         return render_template("home.html",name=names)
      else:
         return render_template("old.html")   


   else:
      names.clear()
      for i in range(8):
        results =requests.get("http://www.omdbapi.com/?apikey=629b6ae&t="+(popular_movies.iloc[i].original_title)).json()
        names.append({'poster':results['Poster'],
                      'title':results['Title'],
                      'overview':results['Plot'],
                      'rating':results['imdbRating'],
                      'votes':results['imdbVotes'],
                      'genre':results['Genre'],
                      'date':results['Released'],
                      'actors':results['Actors'],
                      'director':results['Director']})
      return render_template("new1.html",name =names)
     
@app.route('/<name>', methods=['GET']) 
def foo(name):
   moviename = name
   recommend(moviename)
   return render_template("home.html",name=names)

   


if __name__ == '__main__':
   app.run(debug = True)