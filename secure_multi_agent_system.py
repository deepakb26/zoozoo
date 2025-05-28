import json
from multi_agent_system import MultiAgentSystem
from guardrails_integration import GuardrailsManager

class SecureMultiAgentSystem:
    """
    A secure multi-agent system that applies guardrails to all interactions.
    """
    
    def __init__(self, 
                 supervisor_agent_id=None, 
                 supervisor_agent_alias_id=None,
                 model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                 region_name="us-east-1",
                 s3_bucket=None,
                 sns_topic_arn=None,
                 guardrail_id=None,
                 guardrail_version=None):
        """
        Initialize the secure multi-agent system.
        
        Args:
            supervisor_agent_id (str): Bedrock agent ID for the supervisor
            supervisor_agent_alias_id (str): Bedrock agent alias ID for the supervisor
            model_id (str): Bedrock model ID to use for sub-agents
            region_name (str): AWS region name
            s3_bucket (str): S3 bucket containing the knowledge base
            sns_topic_arn (str): ARN of the SNS topic for emergency notifications
            guardrail_id (str): ID of the Bedrock guardrail
            guardrail_version (str): Version of the Bedrock guardrail
        """
        # Initialize the multi-agent system
        self.multi_agent_system = MultiAgentSystem(
            supervisor_agent_id=supervisor_agent_id,
            supervisor_agent_alias_id=supervisor_agent_alias_id,
            model_id=model_id,
            region_name=region_name,
            s3_bucket=s3_bucket,
            sns_topic_arn=sns_topic_arn
        )
        
        # Initialize the guardrails manager
        self.guardrails = GuardrailsManager(
            guardrail_id=guardrail_id,
            guardrail_version=guardrail_version,
            region_name=region_name
        )
        
        self.model_id = model_id
    
    def process_request(self, user_input, session_id=None):
        """
        Process a user request with guardrails applied.
        
        Args:
            user_input (str): User's message
            session_id (str, optional): Session ID for continuing a conversation
            
        Returns:
            dict: Response with guardrails applied
        """
        # Apply input guardrails
        input_guardrail_result = self.guardrails.apply_guardrails(user_input, self.model_id)
        
        # If input is blocked by guardrails, return the blocked response
        if input_guardrail_result.get('is_blocked', False):
            return {
                'status': 'blocked_by_guardrails',
                'message': "I'm sorry, but I cannot process this request as it violates our content policies.",
                'blocked_reasons': input_guardrail_result.get('blocked_reasons', [])
            }
        
        # Process the filtered input with the multi-agent system
        filtered_input = input_guardrail_result.get('filtered_input', user_input)
        response = self.multi_agent_system.process_request(filtered_input, session_id)
        
        # Extract the response text to apply output guardrails
        response_text = self._extract_response_text(response)
        
        # Apply output guardrails
        output_guardrail_result = self.guardrails.apply_guardrails_to_output(response_text, self.model_id)
        
        # If output is blocked by guardrails, return a safe response
        if output_guardrail_result.get('is_blocked', False):
            return {
                'status': 'output_blocked_by_guardrails',
                'message': "I'm sorry, but I cannot provide the generated response as it violates our content policies.",
                'blocked_reasons': output_guardrail_result.get('blocked_reasons', [])
            }
        
        # Update the response with the filtered output
        filtered_response = self._update_response_text(response, output_guardrail_result.get('filtered_output', response_text))
        
        return filtered_response
    
    def _extract_response_text(self, response):
        """
        Extract the text content from a response object.
        
        Args:
            response (dict): Response from the multi-agent system
            
        Returns:
            str: Text content of the response
        """
        if isinstance(response, str):
            return response
            
        if isinstance(response, dict):
            # Check common response fields
            if 'message' in response:
                return response['message']
            elif 'answer' in response:
                return response['answer']
            elif 'response' in response:
                return response['response']
            
            # Convert the entire response to a string as fallback
            return json.dumps(response)
        
        return str(response)
    
    def _update_response_text(self, response, filtered_text):
        """
        Update the response object with filtered text.
        
        Args:
            response (dict): Original response from the multi-agent system
            filtered_text (str): Filtered text from guardrails
            
        Returns:
            dict: Updated response with filtered text
        """
        if isinstance(response, str):
            return filtered_text
            
        if isinstance(response, dict):
            updated_response = response.copy()
            
            # Update common response fields
            if 'message' in updated_response:
                updated_response['message'] = filtered_text
            elif 'answer' in updated_response:
                updated_response['answer'] = filtered_text
            elif 'response' in updated_response:
                updated_response['response'] = filtered_text
            else:
                # Add a new field if none of the expected fields exist
                updated_response['filtered_response'] = filtered_text
            
            return updated_response
        
        return {'response': filtered_text}