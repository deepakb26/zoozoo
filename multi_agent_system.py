import json
import os
from supervisor_integration import SupervisorAgentClient
from bedrock_integration import BedrockTicketingAgent
from faq_agent import FAQAgent
from emergency_agent import EmergencyAgent

class MultiAgentSystem:
    """
    A multi-agent collaboration system that coordinates between different specialized agents.
    """
    
    def __init__(self, 
                 supervisor_agent_id=None, 
                 supervisor_agent_alias_id=None,
                 model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                 region_name="us-east-1",
                 s3_bucket=None,
                 sns_topic_arn=None):
        """
        Initialize the multi-agent system.
        
        Args:
            supervisor_agent_id (str): Bedrock agent ID for the supervisor
            supervisor_agent_alias_id (str): Bedrock agent alias ID for the supervisor
            model_id (str): Bedrock model ID to use for sub-agents
            region_name (str): AWS region name
            s3_bucket (str): S3 bucket containing the knowledge base
            sns_topic_arn (str): ARN of the SNS topic for emergency notifications
        """
        # Initialize the supervisor agent
        self.supervisor = SupervisorAgentClient(
            agent_id=supervisor_agent_id,
            agent_alias_id=supervisor_agent_alias_id,
            region_name=region_name
        )
        
        # Initialize sub-agents
        self.ticketing_agent = BedrockTicketingAgent(
            model_id=model_id,
            region_name=region_name
        )
        
        self.faq_agent = FAQAgent(
            model_id=model_id,
            region_name=region_name,
            s3_bucket=s3_bucket
        )
        
        self.emergency_agent = EmergencyAgent(
            model_id=model_id,
            region_name=region_name,
            sns_topic_arn=sns_topic_arn
        )
    
    def process_request(self, user_input, session_id=None):
        """
        Process a user request by routing it to the appropriate agent.
        
        Args:
            user_input (str): User's message
            session_id (str, optional): Session ID for continuing a conversation
            
        Returns:
            dict: Response from the appropriate agent
        """
        # First, check if it's an emergency
        emergency_keywords = ["emergency", "accident", "injury", "fire", "urgent", "help", "danger"]
        if any(keyword in user_input.lower() for keyword in emergency_keywords):
            emergency_evaluation = self.emergency_agent.evaluate_emergency(user_input)
            if emergency_evaluation.get('is_emergency', False):
                return self.emergency_agent.handle_emergency_request(user_input)
        
        # If not an emergency, ask the supervisor to determine the intent
        supervisor_response = self.supervisor.invoke_agent(user_input, session_id)
        
        # Extract the routing decision from the supervisor's response
        routing_decision = self._extract_routing_decision(supervisor_response)
        
        # Route to the appropriate agent based on the supervisor's decision
        if routing_decision.get('agent') == 'ticketing':
            return self.ticketing_agent.process_request(user_input)
        elif routing_decision.get('agent') == 'faq':
            return self.faq_agent.answer_question(user_input)
        elif routing_decision.get('agent') == 'emergency':
            return self.emergency_agent.handle_emergency_request(user_input)
        else:
            # Default to supervisor's response if no specific routing
            return {
                'response': self._extract_supervisor_message(supervisor_response),
                'session_id': supervisor_response.get('sessionId')
            }
    
    def _extract_routing_decision(self, supervisor_response):
        """
        Extract the routing decision from the supervisor's response.
        
        Args:
            supervisor_response (dict): Response from the supervisor agent
            
        Returns:
            dict: Routing decision with agent and any additional parameters
        """
        # Default decision
        decision = {'agent': 'supervisor'}
        
        # Try to find routing information in the supervisor's response
        messages = supervisor_response.get('messages', [])
        for message in messages:
            content = message.get('content', '')
            
            # Look for JSON routing information
            try:
                if '{"agent":' in content:
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = content[start_idx:end_idx]
                        routing_info = json.loads(json_str)
                        if 'agent' in routing_info:
                            return routing_info
            except json.JSONDecodeError:
                pass
            
            # Simple keyword-based routing as fallback
            content_lower = content.lower()
            if 'ticket' in content_lower or 'create ticket' in content_lower or 'cancel ticket' in content_lower:
                decision['agent'] = 'ticketing'
            elif 'question' in content_lower or 'faq' in content_lower or 'knowledge base' in content_lower:
                decision['agent'] = 'faq'
            elif 'emergency' in content_lower or 'urgent' in content_lower:
                decision['agent'] = 'emergency'
        
        return decision
    
    def _extract_supervisor_message(self, supervisor_response):
        """
        Extract the main message from the supervisor's response.
        
        Args:
            supervisor_response (dict): Response from the supervisor agent
            
        Returns:
            str: Main message from the supervisor
        """
        messages = supervisor_response.get('messages', [])
        if messages:
            return messages[0].get('content', '')
        return ''