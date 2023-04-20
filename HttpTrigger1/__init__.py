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

def articles_clickes_by_user(data ,user_id):
    article_du_user = data.loc[data["user_id"] == user_id]
    return article_du_user

def articles_recommandation(article_id, indices_article, sim_cosine):
    idx = indices_article[article_id]
    sim_scores = list(enumerate(sim_cosine[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    return sim_scores

def user_recommandation(user_id, indices_article, sim_cosine, data):
    logging.info('------------debut  traitement3')
    reco = [] # Création d'un tableau vide 
    articles_du_user_clickes = articles_clickes_by_user(data, user_id) # Récupération des articles qui ont été cliquers par un certain utilisateur
    logging.info('------------fin  traitement3')
    logging.info('------------debut  traitement4')
    for art in list(articles_du_user_clickes["article_id"]): # Boucle sur les articles qui ont été cliqué par cet utilisateur
        article_recommande = articles_recommandation(art, indices_article, sim_cosine) # Cette fonction récupère les articles les plus similaires
        reco.append(article_recommande[0]) # Dans le tableau reco les articles les plus similaires y sont ajoutés
    logging.info('------------fin  traitement4')
    logging.info('------------debut  traitement5')
    sim_scores = sorted(reco, key=lambda x: x[1], reverse=True) # trie les articles de facon a ce que le plus recommandé soit présent dans les 5 articles les plus recommandé
    sim_scores = sim_scores[1:6] # Récupère les 5 articles les plus recommmandé
    logging.info('------------fin  traitement5')
    logging.info('------------debut  traitement6')
    json_result = []
    for item in sim_scores:
        json_result.append({"article_id_recommandé":item[0], "score_cosine_de_similarité":str(item[1])})
    logging.info('------------fin  traitement6'+str(json_result))
    return json_result

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


    dataframe_all_clicks = df.join(articles, how='left', on='click_article_id', lsuffix="_1")

    # Créer une map pour convertir les article_id en category
    dict_article_categories = articles.set_index('article_id')['category_id'].to_dict()

    # Récupère les catégories associées pour chaque article
    dataframe_all_clicks['category_id'] = dataframe_all_clicks['click_article_id'].map(dict_article_categories).astype(int)
    dataframe_all_clicks['total_click'] = dataframe_all_clicks.groupby(['user_id'])['click_article_id'].transform('count')
    dataframe_all_clicks['total_click_by_category_id'] = dataframe_all_clicks.groupby(['user_id','category_id'])['click_article_id'].transform('count')
    dataframe_all_clicks['rating'] = dataframe_all_clicks['total_click_by_category_id'] / dataframe_all_clicks['total_click']

    dataframe_all_clicks = dataframe_all_clicks.drop_duplicates()

    dataframe_all_clicks = dataframe_all_clicks.drop(['total_click', 'total_click_by_category_id'], axis=1)

    dataframe_all_clicks_sav = dataframe_all_clicks.copy()

    dataframe_all_clicks = dataframe_all_clicks.drop(['click_article_id'], axis=1)

    articles_clicks_once = dataframe_all_clicks_sav.click_article_id.value_counts().index

    df_articles_clicks_once = pd.DataFrame(articles_clicks_once).reset_index() 

    df_articles_clicks_once.columns = ["index", "article_id"]

    

    # La jointure est faite sur les articles qui ont étés cliqués au moins une fois et sur l'embedding réduit par l'ACP, la jointure se fait sur l'index
    dataframe_arts_embedd_acp = df_articles_clicks_once.join(pca_tranformed_embedding_df, how='left', on='index', lsuffix="_1")
    
    user_table = (articles_clickes_by_user(dataframe_all_clicks_sav, user_id)).reset_index()

    # suppression des articles déjà cliqués par le user
    df_arts_embedd_acp_filt = dataframe_arts_embedd_acp[~dataframe_arts_embedd_acp.article_id.isin(user_table.article_id)]
    articles_embedd_acp = df_arts_embedd_acp_filt.iloc[:, 3:]

    user_table = user_table[["user_id", "index", "article_id"]]

    dataframe_user_merged = user_table.merge(dataframe_arts_embedd_acp, how='left', on='article_id')
    # Les tables qui sont merge sont les cliques de l'utilisateur 0 et l'embedding réduit par l'ACP

    dataframe_user_merged = dataframe_user_merged.iloc[:,5:]
    logging.info('------------fin traitement')
    logging.info('------------debut cosinus traitement')
    sim_cosine = cosine_similarity(dataframe_user_merged, articles_embedd_acp)
    logging.info('------------fin cosinus traitement')
    logging.info('------------debut  traitement1')
    titles_article = user_table["article_id"]
    indices_article = pd.Series(range(0,len(user_table["article_id"])), index=titles_article)
    logging.info('------------fin  traitement1')
    return user_recommandation(user_id, indices_article, sim_cosine, dataframe_all_clicks_sav)


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
    logging.info('------------get variable debut')
    req_body_bytes = req.get_body()
    req_body = req_body_bytes.decode("utf-8")
    json_body = json.loads(req_body)
    name = None
    name = json_body['user_id']
    logging.info('------------recupere variable ok')
    func.HttpResponse.charset = 'utf-8'
    name = int(name)
    return func.HttpResponse(
            json.dumps(recommandation_generator(pca_tranformed_embedding_df, articles, df, name)),
            status_code=200
            )
