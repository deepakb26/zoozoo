import unittest
from unittest.mock import patch, MagicMock
from emergency_agent import EmergencyAgent
import json

class TestEmergencyAgent(unittest.TestCase):
    """Test cases for the EmergencyAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('boto3.client') as mock_client:
            self.mock_bedrock_runtime = MagicMock()
            self.mock_sns = MagicMock()
            
            # Configure the mock to return different clients
            def get_client(service_name, region_name):
                if service_name == 'bedrock-runtime':
                    return self.mock_bedrock_runtime
                elif service_name == 'sns':
                    return self.mock_sns
            
            mock_client.side_effect = get_client
            
            self.agent = EmergencyAgent(sns_topic_arn='test-topic-arn')
    
    def test_evaluate_emergency_high_severity(self):
        """Test evaluating a high severity emergency."""
        # Mock the Bedrock response
        mock_response = {
            'body': MagicMock()
        }
        mock_response['body'].read.return_value = json.dumps({
            'content': [
                {
                    'text': '{"is_emergency": true, "severity": "high", "recommended_actions": ["Call 911", "Evacuate the building"], "reasoning": "This is a serious fire situation"}'
                }
            ]
        })
        self.mock_bedrock_runtime.invoke_model.return_value = mock_response
        
        # Test the evaluation
        result = self.agent.evaluate_emergency("There's a fire in the building and it's spreading quickly!")
        
        # Verify the result
        self.assertTrue(result['is_emergency'])
        self.assertEqual(result['severity'], 'high')
        self.assertIn('Call 911', result['recommended_actions'])
        
        # Verify SNS was called for escalation
        self.mock_sns.publish.assert_called_once()
        args, kwargs = self.mock_sns.publish.call_args
        self.assertEqual(kwargs['TopicArn'], 'test-topic-arn')
        self.assertIn('EMERGENCY ALERT', kwargs['Subject'])
    
    def test_evaluate_emergency_low_severity(self):
        """Test evaluating a low severity situation."""
        # Mock the Bedrock response
        mock_response = {
            'body': MagicMock()
        }
        mock_response['body'].read.return_value = json.dumps({
            'content': [
                {
                    'text': '{"is_emergency": false, "severity": "low", "recommended_actions": ["Monitor the situation"], "reasoning": "This is a minor issue"}'
                }
            ]
        })
        self.mock_bedrock_runtime.invoke_model.return_value = mock_response
        
        # Test the evaluation
        result = self.agent.evaluate_emergency("The printer is out of paper.")
        
        # Verify the result
        self.assertFalse(result['is_emergency'])
        self.assertEqual(result['severity'], 'low')
        
        # Verify SNS was NOT called for escalation
        self.mock_sns.publish.assert_not_called()
    
    def test_handle_emergency_request_critical(self):
        """Test handling a critical emergency request."""
        # Mock the evaluate_emergency method
        with patch.object(self.agent, 'evaluate_emergency') as mock_evaluate:
            mock_evaluate.return_value = {
                'is_emergency': True,
                'severity': 'critical',
                'recommended_actions': ['Call 911', 'Administer first aid'],
                'reasoning': 'Medical emergency requiring immediate attention'
            }
            
            # Test handling the request
            result = self.agent.handle_emergency_request("Someone collapsed and isn't breathing!")
            
            # Verify the result
            self.assertEqual(result['status'], 'emergency_escalated')
            self.assertIn('serious emergency', result['message'])
            self.assertEqual(len(result['actions']), 2)
            self.assertEqual(result['severity'], 'critical')

if __name__ == '__main__':
    unittest.main()