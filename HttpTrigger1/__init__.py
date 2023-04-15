import logging
import pandas as pd
import json
import azure.functions as func
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

def outlier(col):
    sorted(col)
    Q1, Q3 = col.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    Q1min = Q1 - (1.5 * IQR)
    Q3max = Q3 + (1.5 * IQR)
    return Q1min, Q3max

def user_articles_clickes(data ,user_id):
    article_du_user = data.loc[data["user_id"] == user_id]
    return article_du_user

def articles_reco(article_id, indices, cosine_sim2):
    idx = indices[article_id]
    sim_scores = list(enumerate(cosine_sim2[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    return sim_scores

def user_reco(user_id, indices, cosine_sim2, data):
    reco = [] # Création d'un tableau vide 
    articles_du_user_clickes = user_articles_clickes(data, user_id) # Récupération des articles qui ont été cliquers par un certain utilisateur
    for art in list(articles_du_user_clickes["article_id"]): # Boucle sur les articles qui ont été cliqué par cet utilisateur
        article_recommande = articles_reco(art, indices, cosine_sim2) # Cette fonction récupère les articles les plus similaires
        reco.append(article_recommande[0]) # Dans le tableau reco les articles les plus similaires y sont ajoutés
    sim_scores = sorted(reco, key=lambda x: x[1], reverse=True) # trie les articles de facon a ce que le plus recommandé soit présent dans les 5 articles les plus recommandé
    sim_scores = sim_scores[1:6] # Récupère les 5 articles les plus recommmandé
    return json.dumps(sim_scores)

def transform_to_dataframecsv(blob):
    dfs = bytearray(blob.read())
    dfs = pd.read_csv(BytesIO(dfs))
    return dfs

def recommandation_generator(pca_tranformed_embedding_df, articles,  df, user_id):
    logging.info('-----------------------debut fonction recommandation')
    logging.info('------------debut traitement')

    minScore, maxScore = outlier(articles["words_count"])

    articles["words_count"] = np.where(articles["words_count"] < minScore, minScore, articles["words_count"])
    articles["words_count"] = np.where(articles["words_count"] > maxScore, maxScore, articles["words_count"])


    all_clicks_df = df.join(articles, how='left', on='click_article_id', lsuffix="_1")

    #Create a map to convert article_id to category
    dict_article_categories = articles.set_index('article_id')['category_id'].to_dict()

    #Get Categorie associate for each article
    all_clicks_df['category_id'] = all_clicks_df['click_article_id'].map(dict_article_categories).astype(int)
    all_clicks_df['total_click'] = all_clicks_df.groupby(['user_id'])['click_article_id'].transform('count')
    all_clicks_df['total_click_by_category_id'] = all_clicks_df.groupby(['user_id','category_id'])['click_article_id'].transform('count')
    all_clicks_df['rating'] = all_clicks_df['total_click_by_category_id'] / all_clicks_df['total_click']

    all_clicks_df = all_clicks_df.drop_duplicates()

    all_clicks_df = all_clicks_df.drop(['total_click', 'total_click_by_category_id'], axis=1)

    all_clicks_df_sav = all_clicks_df.copy()

    all_clicks_df = all_clicks_df.drop(['click_article_id'], axis=1)

    articles_click_au_moins_une_fois = all_clicks_df_sav.click_article_id.value_counts().index

    df_articles_click_au_moins_une_fois = pd.DataFrame(articles_click_au_moins_une_fois).reset_index() 

    df_articles_click_au_moins_une_fois.columns =["index", "article_id"]

    

    # La jointure est faite sur les articles qui ont étés cliqués au moins une fois et sur l'embedding réduit par l'ACP, la jointure se fait sur l'index
    df_arts_embedd_acp = df_articles_click_au_moins_une_fois.join(pca_tranformed_embedding_df, how='left', on='index', lsuffix="_1")

    arts_embedd_acp = df_arts_embedd_acp.iloc[:, 3:]

    table_user = (user_articles_clickes(all_clicks_df_sav, 0)).reset_index()

    table_user = table_user[["user_id", "index", "article_id"]]

    df_user_merged = table_user.merge(df_arts_embedd_acp, how='left', on='article_id')
    # Les tables qui sont merge sont les cliques de l'utilisateur 0 et l'embedding réduit par l'ACP

    df_user_merged = df_user_merged.iloc[:,5:]
    logging.info('------------fin traitement')
    logging.info('------------debut cosinus traitement')
    cosine_sim2 = cosine_similarity(df_user_merged, arts_embedd_acp)
    logging.info('------------fin cosinus traitement')
    titles = table_user["article_id"]
    indices = pd.Series(range(0,8), index=titles)
    return user_reco(user_id, indices, cosine_sim2, all_clicks_df_sav)


def main(req: func.HttpRequest, dfpcablob: func.InputStream, dfartblob: func.InputStream, dfblob: func.InputStream) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    logging.info('------------debut fichier acp')
    pca_tranformed_embedding_df = transform_to_dataframecsv(dfpcablob) # acp.csv
    logging.info('------------fin fichier acp')
    logging.info('------------debut fichier art')
    articles = transform_to_dataframecsv(dfartblob)#articles_metadata.csv
    logging.info('------------fin fichier art')
    logging.info('------------debut fichier df')
    df = transform_to_dataframecsv(dfblob)  # df_clicks.csv
    logging.info('------------fin fichier df')
    name = req.params.get('user_id')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('user_id')

    if name:
        return func.HttpResponse(recommandation_generator(pca_tranformed_embedding_df, articles,  df,name),
                status_code=200)
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
