import unittest
from unittest.mock import patch, MagicMock
from faq_agent import FAQAgent
import json

class TestFAQAgent(unittest.TestCase):
    """Test cases for the FAQAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('boto3.client') as mock_client:
            self.mock_bedrock_runtime = MagicMock()
            self.mock_s3 = MagicMock()
            
            # Configure the mock to return different clients
            def get_client(service_name, region_name):
                if service_name == 'bedrock-runtime':
                    return self.mock_bedrock_runtime
                elif service_name == 's3':
                    return self.mock_s3
            
            mock_client.side_effect = get_client
            
            self.agent = FAQAgent(s3_bucket='test-bucket')
    
    def test_retrieve_relevant_documents_found(self):
        """Test retrieving relevant documents when matches are found."""
        # Mock the S3 list_objects_v2 response
        self.mock_s3.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'knowledge_base/doc1.txt'},
                {'Key': 'knowledge_base/doc2.md'},
                {'Key': 'knowledge_base/doc3.json'}
            ]
        }
        
        # Mock the S3 get_object responses
        def mock_get_object(Bucket, Key):
            if Key == 'knowledge_base/doc1.txt':
                mock_body = MagicMock()
                mock_body.read.return_value = b'This is a document about refunds and returns policy'
                return {'Body': mock_body}
            elif Key == 'knowledge_base/doc2.md':
                mock_body = MagicMock()
                mock_body.read.return_value = b'Information about shipping and delivery times'
                return {'Body': mock_body}
            else:
                mock_body = MagicMock()
                mock_body.read.return_value = b'{"topic": "product warranty information"}'
                return {'Body': mock_body}
        
        self.mock_s3.get_object.side_effect = mock_get_object
        
        # Test retrieving documents
        docs = self.agent.retrieve_relevant_documents("What is your refund policy?")
        
        # Verify the results
        self.assertEqual(len(docs), 1)
        self.assertIn('refunds', docs[0]['content'].lower())
        self.assertEqual(docs[0]['source'], 'knowledge_base/doc1.txt')
    
    def test_retrieve_relevant_documents_not_found(self):
        """Test retrieving relevant documents when no matches are found."""
        # Mock the S3 list_objects_v2 response
        self.mock_s3.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'knowledge_base/doc1.txt'},
                {'Key': 'knowledge_base/doc2.md'}
            ]
        }
        
        # Mock the S3 get_object responses
        def mock_get_object(Bucket, Key):
            if Key == 'knowledge_base/doc1.txt':
                mock_body = MagicMock()
                mock_body.read.return_value = b'This is a document about refunds and returns policy'
                return {'Body': mock_body}
            else:
                mock_body = MagicMock()
                mock_body.read.return_value = b'Information about shipping and delivery times'
                return {'Body': mock_body}
        
        self.mock_s3.get_object.side_effect = mock_get_object
        
        # Test retrieving documents with no matches
        docs = self.agent.retrieve_relevant_documents("What are your store hours?")
        
        # Verify no documents were found
        self.assertEqual(len(docs), 0)
    
    def test_answer_question_with_context(self):
        """Test answering a question with relevant context."""
        # Mock document retrieval
        with patch.object(self.agent, 'retrieve_relevant_documents') as mock_retrieve:
            mock_retrieve.return_value = [
                {
                    'content': 'Our refund policy allows returns within 30 days of purchase with receipt.',
                    'source': 'knowledge_base/refunds.txt',
                    'relevance': 2
                }
            ]
            
            # Mock the Bedrock response
            mock_response = {
                'body': MagicMock()
            }
            mock_response['body'].read.return_value = json.dumps({
                'content': [
                    {
                        'text': 'You can return items within 30 days of purchase if you have the receipt.'
                    }
                ]
            })
            self.mock_bedrock_runtime.invoke_model.return_value = mock_response
            
            # Test answering the question
            result = self.agent.answer_question("What is your return policy?")
            
            # Verify the result
            self.assertIn('30 days', result['answer'])
            self.assertEqual(len(result['sources']), 1)
            self.assertEqual(result['sources'][0], 'knowledge_base/refunds.txt')
    
    def test_answer_question_no_context(self):
        """Test answering a question with no relevant context."""
        # Mock document retrieval with no results
        with patch.object(self.agent, 'retrieve_relevant_documents') as mock_retrieve:
            mock_retrieve.return_value = []
            
            # Test answering the question
            result = self.agent.answer_question("What is your store hours?")
            
            # Verify the result indicates no information was found
            self.assertIn("couldn't find", result['answer'].lower())
            self.assertEqual(len(result['sources']), 0)
            
            # Verify Bedrock was not called
            self.mock_bedrock_runtime.invoke_model.assert_not_called()

if __name__ == '__main__':
    unittest.main()