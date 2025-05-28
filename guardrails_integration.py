import boto3
import json

class GuardrailsManager:
    """
    A manager for integrating AWS Bedrock Guardrails with the multi-agent system.
    """
    
    def __init__(self, guardrail_id=None, guardrail_version=None, region_name="us-east-1"):
        """
        Initialize the guardrails manager.
        
        Args:
            guardrail_id (str): ID of the Bedrock guardrail
            guardrail_version (str): Version of the Bedrock guardrail
            region_name (str): AWS region name
        """
        self.bedrock = boto3.client('bedrock', region_name=region_name)
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region_name)
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version
    
    def apply_guardrails(self, input_text, model_id):
        """
        Apply guardrails to input text before sending to a model.
        
        Args:
            input_text (str): User input text
            model_id (str): Bedrock model ID
            
        Returns:
            dict: Result of guardrail application with filtered input
        """
        if not self.guardrail_id or not self.guardrail_version:
            # If guardrails not configured, return original input
            return {
                'is_blocked': False,
                'filtered_input': input_text,
                'blocked_reasons': []
            }
        
        try:
            response = self.bedrock_runtime.apply_guardrail(
                guardrailIdentifier=self.guardrail_id,
                guardrailVersion=self.guardrail_version,
                contentType='application/json',
                accept='application/json',
                body=json.dumps({
                    'input': input_text
                })
            )
            
            result = json.loads(response['body'].read())
            
            return {
                'is_blocked': result.get('assessment', {}).get('topicPolicy', {}).get('blocked', False),
                'filtered_input': result.get('output', input_text),
                'blocked_reasons': result.get('assessment', {}).get('topicPolicy', {}).get('topics', [])
            }
            
        except Exception as e:
            print(f"Error applying guardrails: {e}")
            # On error, return original input
            return {
                'is_blocked': False,
                'filtered_input': input_text,
                'blocked_reasons': []
            }
    
    def apply_guardrails_to_output(self, output_text, model_id):
        """
        Apply guardrails to model output text.
        
        Args:
            output_text (str): Model output text
            model_id (str): Bedrock model ID
            
        Returns:
            dict: Result of guardrail application with filtered output
        """
        if not self.guardrail_id or not self.guardrail_version:
            # If guardrails not configured, return original output
            return {
                'is_blocked': False,
                'filtered_output': output_text,
                'blocked_reasons': []
            }
        
        try:
            response = self.bedrock_runtime.apply_guardrail(
                guardrailIdentifier=self.guardrail_id,
                guardrailVersion=self.guardrail_version,
                contentType='application/json',
                accept='application/json',
                body=json.dumps({
                    'output': output_text
                })
            )
            
            result = json.loads(response['body'].read())
            
            return {
                'is_blocked': result.get('assessment', {}).get('topicPolicy', {}).get('blocked', False),
                'filtered_output': result.get('output', output_text),
                'blocked_reasons': result.get('assessment', {}).get('topicPolicy', {}).get('topics', [])
            }
            
        except Exception as e:
            print(f"Error applying guardrails to output: {e}")
            # On error, return original output
            return {
                'is_blocked': False,
                'filtered_output': output_text,
                'blocked_reasons': []
            }
    
    def get_guardrail_details(self):
        """
        Get details about the configured guardrail.
        
        Returns:
            dict: Guardrail details
        """
        if not self.guardrail_id:
            return {'error': 'No guardrail ID configured'}
        
        try:
            response = self.bedrock.get_guardrail(
                guardrailIdentifier=self.guardrail_id,
                guardrailVersion=self.guardrail_version if self.guardrail_version else '$LATEST'
            )
            
            return {
                'name': response.get('name'),
                'version': response.get('version'),
                'status': response.get('status'),
                'description': response.get('description'),
                'blockedTopics': response.get('blockedTopics', []),
                'contentPolicyConfig': response.get('contentPolicyConfig', {})
            }
            
        except Exception as e:
            print(f"Error getting guardrail details: {e}")
            return {'error': str(e)}