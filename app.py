from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from flasgger import Swagger
from sentence_transformers import SentenceTransformer, util


app = Flask(__name__)
api = Api(app)
swagger = Swagger(app)
model = SentenceTransformer("all-MiniLM-L6-v2")

def map_to_scale(x):
    return (x + 1) / 2


def SimilarityScore(text_1, text_2):
        '''
        This function is used to calculate the Semantic Textual Similarity between the two paragraphs. 
        Returns the cosine similarity score.
        '''
        
        embeddings1 = model.encode(text_1, convert_to_tensor=True)
        embeddings2 = model.encode(text_2, convert_to_tensor=True)
        
        cosine_score = util.cos_sim(embeddings1, embeddings2)
        cosine_score = float(cosine_score)
        cosine_score = map_to_scale(cosine_score)
        return cosine_score
        
class CSM(Resource):
    def get(self):
        """
        This method responds to the GET request for this endpoint and calculates the similarity score between two texts.
        ---
        tags:
        - Text Processing
        parameters:
            - name: text1
              in: query
              type: string
              required: true
              description: The first text
            - name: text2
              in: query
              type: string
              required: true
              description: The second text
        responses:
            200:
                description: A successful GET request
                content:
                    application/json:
                      schema:
                        type: object
                        properties:
                            similarityscore:
                                type: float
                                description: The similarity score between the two texts
        """
        text_1= request.args.get('text1')
       
        text_2 =request.args.get('text2')
        text_1 = str(text_1)
        text_2 = str(text_2)
        if text_1 == None or text_2 == None:
            return 500
        
        similarityscore = SimilarityScore(text_1, text_2)
        
        return {'similarity score': similarityscore}, 200
        

api.add_resource(CSM, "/calculate-similarity")

if __name__=="__main__":
    app.run(debug=True, port=8000)