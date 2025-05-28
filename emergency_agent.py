import boto3
import json
from datetime import datetime

class EmergencyAgent:
    """
    An agent that evaluates emergency situations and escalates when necessary.
    """
    
    def __init__(self, model_id="anthropic.claude-3-sonnet-20240229-v1:0", 
                 region_name="us-east-1",
                 sns_topic_arn=None):
        """
        Initialize the emergency agent.
        
        Args:
            model_id (str): Bedrock model ID to use
            region_name (str): AWS region name
            sns_topic_arn (str): ARN of the SNS topic for emergency notifications
        """
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region_name)
        self.sns = boto3.client('sns', region_name=region_name)
        self.model_id = model_id
        self.sns_topic_arn = sns_topic_arn
    
    def evaluate_emergency(self, message):
        """
        Evaluate if a message describes an emergency situation.
        
        Args:
            message (str): User's message describing a situation
            
        Returns:
            dict: Evaluation results including severity level and recommended actions
        """
        prompt = f"""
        You are an emergency evaluation system. Analyze the following situation and determine:
        1. If it's an emergency
        2. The severity level (low, medium, high, critical)
        3. What immediate actions should be taken
        
        Return a JSON object with the following structure:
        {{
            "is_emergency": true/false,
            "severity": "low|medium|high|critical",
            "recommended_actions": ["action1", "action2", ...],
            "reasoning": "brief explanation of your assessment"
        }}
        
        Situation: {message}
        
        JSON response:
        """
        
        # Call Bedrock model to evaluate the emergency
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
            content = response_body.get('content', [{}])[0].get('text', '{}')
            
            # Extract JSON from the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                evaluation = json.loads(json_str)
                
                # If it's a high or critical emergency, escalate
                if evaluation.get('is_emergency', False) and evaluation.get('severity') in ['high', 'critical']:
                    self._escalate_emergency(message, evaluation)
                
                return evaluation
            
            return {
                "is_emergency": False,
                "severity": "unknown",
                "recommended_actions": [],
                "reasoning": "Failed to evaluate the situation"
            }
            
        except Exception as e:
            print(f"Error evaluating emergency: {e}")
            return {
                "is_emergency": False,
                "severity": "unknown",
                "recommended_actions": [],
                "reasoning": f"Error: {str(e)}"
            }
    
    def _escalate_emergency(self, message, evaluation):
        """
        Escalate an emergency by sending notifications.
        
        Args:
            message (str): Original emergency message
            evaluation (dict): Emergency evaluation results
        """
        if not self.sns_topic_arn:
            print("Warning: SNS topic ARN not configured. Emergency notification not sent.")
            return
        
        try:
            # Prepare the notification message
            notification = {
                "timestamp": datetime.utcnow().isoformat(),
                "original_message": message,
                "severity": evaluation.get('severity'),
                "recommended_actions": evaluation.get('recommended_actions', []),
                "reasoning": evaluation.get('reasoning', '')
            }
            
            # Send the notification
            self.sns.publish(
                TopicArn=self.sns_topic_arn,
                Subject=f"EMERGENCY ALERT - {evaluation.get('severity', 'high')} severity",
                Message=json.dumps(notification, indent=2)
            )
            
            print(f"Emergency escalated: {evaluation.get('severity')} severity")
            
        except Exception as e:
            print(f"Error escalating emergency: {e}")
    
    def handle_emergency_request(self, message):
        """
        Handle an emergency request from a user.
        
        Args:
            message (str): User's message describing a potential emergency
            
        Returns:
            dict: Response with evaluation and next steps
        """
        # Evaluate the emergency
        evaluation = self.evaluate_emergency(message)
        
        # Prepare response based on severity
        if evaluation.get('is_emergency', False):
            if evaluation.get('severity') in ['high', 'critical']:
                response = {
                    'status': 'emergency_escalated',
                    'message': "This situation has been identified as a serious emergency and has been escalated to the appropriate team. Please follow these immediate actions:",
                    'actions': evaluation.get('recommended_actions', []),
                    'severity': evaluation.get('severity')
                }
            else:
                response = {
                    'status': 'emergency_identified',
                    'message': "This situation has been identified as a potential emergency. Please consider these recommended actions:",
                    'actions': evaluation.get('recommended_actions', []),
                    'severity': evaluation.get('severity')
                }
        else:
            response = {
                'status': 'not_emergency',
                'message': "This situation has not been identified as an emergency.",
                'reasoning': evaluation.get('reasoning', ''),
                'severity': evaluation.get('severity', 'low')
            }
        
        return response