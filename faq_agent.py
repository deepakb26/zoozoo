import boto3
import json
from botocore.exceptions import ClientError

class FAQAgent:
    """
    A RAG-based FAQ agent that retrieves information from a knowledge base in S3.
    """
    
    def __init__(self, model_id="anthropic.claude-3-sonnet-20240229-v1:0", 
                 region_name="us-east-1", 
                 s3_bucket=None,
                 s3_prefix="knowledge_base/"):
        """
        Initialize the FAQ agent.
        
        Args:
            model_id (str): Bedrock model ID to use
            region_name (str): AWS region name
            s3_bucket (str): S3 bucket containing the knowledge base
            s3_prefix (str): Prefix for knowledge base files in S3
        """
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region_name)
        self.s3 = boto3.client('s3', region_name=region_name)
        self.model_id = model_id
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
    
    def retrieve_relevant_documents(self, query, max_docs=3):
        """
        Retrieve relevant documents from the knowledge base based on the query.
        
        Args:
            query (str): User's question
            max_docs (int): Maximum number of documents to retrieve
            
        Returns:
            list: List of relevant document contents
        """
        try:
            # List objects in the S3 bucket with the specified prefix
            response = self.s3.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=self.s3_prefix
            )
            
            if 'Contents' not in response:
                return []
            
            # Simple keyword-based retrieval (in production, use a vector DB)
            relevant_docs = []
            query_terms = set(query.lower().split())
            
            for obj in response.get('Contents', []):
                if len(relevant_docs) >= max_docs:
                    break
                    
                key = obj['Key']
                if not key.endswith('.txt') and not key.endswith('.md') and not key.endswith('.json'):
                    continue
                
                # Get the document content
                doc_response = self.s3.get_object(Bucket=self.s3_bucket, Key=key)
                content = doc_response['Body'].read().decode('utf-8')
                
                # Simple relevance check (replace with proper embedding similarity in production)
                content_terms = set(content.lower().split())
                overlap = len(query_terms.intersection(content_terms))
                
                if overlap > 0:
                    relevant_docs.append({
                        'content': content,
                        'source': key,
                        'relevance': overlap
                    })
            
            # Sort by relevance
            relevant_docs.sort(key=lambda x: x['relevance'], reverse=True)
            return relevant_docs[:max_docs]
            
        except ClientError as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def answer_question(self, query):
        """
        Answer a question using the knowledge base.
        
        Args:
            query (str): User's question
            
        Returns:
            dict: Response containing the answer and sources
        """
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_documents(query)
        
        if not relevant_docs:
            return {
                'answer': "I couldn't find any relevant information in the knowledge base.",
                'sources': []
            }
        
        # Prepare context from relevant documents
        context = "\n\n".join([f"Document from {doc['source']}:\n{doc['content']}" for doc in relevant_docs])
        sources = [doc['source'] for doc in relevant_docs]
        
        # Create prompt with retrieved context
        prompt = f"""
        You are a helpful assistant answering questions based on the provided knowledge base.
        Use ONLY the information in the following documents to answer the question.
        If the documents don't contain the answer, say "I don't have enough information to answer this question."
        
        KNOWLEDGE BASE DOCUMENTS:
        {context}
        
        QUESTION: {query}
        
        ANSWER:
        """
        
        # Call Bedrock model to generate the answer
        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                })
            )
            
            response_body = json.loads(response.get('body').read())
            answer = response_body.get('content', [{}])[0].get('text', '')
            
            return {
                'answer': answer,
                'sources': sources
            }
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                'answer': "I encountered an error while trying to answer your question.",
                'sources': []
            }