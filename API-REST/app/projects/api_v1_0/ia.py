import ast
import gensim
from gensim.models import Word2Vec

from flask import request, Blueprint, jsonify
from flask_restful import Api

from ...db_conn import conn_controller as conn

ia_v1_0_bp = Blueprint('ia_v1_0_bp', __name__)

api = Api(ia_v1_0_bp)

#------------ Embeddings --------------

model = Word2Vec.load("version3")

@ia_v1_0_bp.route("/vectorize", methods=['GET'])
def get_embedding_similarity():
    """
    Get the numeric vector for the embedding 
    ---
    tags:
      - emebeddings
    parameters:
      - in: query
        name: text
        description: Text to be converted to embedding
        type: string

    responses:
      200:
        embedding: Numeric vector corresponding to the text submited.
    """

    params = {}

    text = request.values.get('text')
    if text is not None:
        params['text'] = text
    
    textEmb = params['text']
    
    tokens = textEmb.split(' ')
    
    embs = []
    for token in tokens:
        emb = model.wv[token]
        embs.append(emb)

    result = sum(embs)/len(tokens)

    return jsonify(result.tolist())

@ia_v1_0_bp.route("/cosine", methods=['GET'])
def get_embedding():
    """
    Get the cosine similarity between to sentences
    ---
    tags:
      - emebeddings
    parameters:
      - in: query
        name: text1
        description: First text
        type: string
      - in: query
        name: text2
        description: Second text
        type: string

    responses:
      200:
        similarity: float value indicating the cosine similarity
    """

    params = {}

    text1 = request.values.get('text1')
    text2 = request.values.get('text2')
    if text1 is not None:
        params['text1'] = text1
    if text2 is not None:
        params['text2'] = text2

    firstSentence = params['text1'].split()
    secondSentence = params['text2'].split()
    
    similarity = model.wv.wmdistance(firstSentence, secondSentence)
    

    return jsonify(similarity)

@ia_v1_0_bp.route("/similarity", methods=['GET'])
def get_embedding_op():
    """
    Get the most similar key for the submitted text
    ---
    tags:
      - emebeddings
    parameters:
      - in: query
        name: text
        description: the text used to find the most simmilar value
        type: string

    responses:
      200:
        similarity: most similar key to the text
    """

    params = {}

    text = request.values.get('text')
    if text is not None:
        params['text'] = text

    sentence = params['text'].split()

    result = model.wv.most_similar(positive=sentence)


    return jsonify(result)

