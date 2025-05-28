import boto3
import json
from ticketing_agent import TicketingAgent

class BedrockTicketingAgent:
    """
    A Bedrock-based agent that handles ticketing operations through natural language.
    This agent integrates with the TicketingAgent to perform operations on DynamoDB.
    """
    
    def __init__(self, model_id="anthropic.claude-3-sonnet-20240229-v1:0", region_name="us-east-1"):
        """
        Initialize the Bedrock ticketing agent.
        
        Args:
            model_id (str): Bedrock model ID to use
            region_name (str): AWS region name
        """
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region_name)
        self.model_id = model_id
        self.ticketing_agent = TicketingAgent()
    
    def process_request(self, user_input):
        """
        Process a natural language request and determine the appropriate ticketing action.
        
        Args:
            user_input (str): User's natural language request
            
        Returns:
            dict: Response containing action taken and relevant information
        """
        # Determine the intent from the user input using Bedrock
        intent = self._determine_intent(user_input)
        
        if intent['action'] == 'create_ticket':
            # Extract ticket details from the intent
            ticket = self.ticketing_agent.create_ticket(
                subject=intent.get('subject', 'No subject provided'),
                description=intent.get('description', 'No description provided'),
                priority=intent.get('priority', 'medium'),
                assigned_to=intent.get('assigned_to')
            )
            return {
                'action': 'create_ticket',
                'success': ticket is not None,
                'ticket': ticket
            }
            
        elif intent['action'] == 'get_ticket_status':
            # Get ticket status
            ticket_id = intent.get('ticket_id')
            if not ticket_id:
                return {
                    'action': 'get_ticket_status',
                    'success': False,
                    'error': 'No ticket ID provided'
                }
                
            ticket = self.ticketing_agent.get_ticket_status(ticket_id)
            return {
                'action': 'get_ticket_status',
                'success': ticket is not None,
                'ticket': ticket
            }
            
        elif intent['action'] == 'cancel_ticket':
            # Cancel ticket
            ticket_id = intent.get('ticket_id')
            if not ticket_id:
                return {
                    'action': 'cancel_ticket',
                    'success': False,
                    'error': 'No ticket ID provided'
                }
                
            success = self.ticketing_agent.cancel_ticket(
                ticket_id=ticket_id,
                reason=intent.get('reason')
            )
            return {
                'action': 'cancel_ticket',
                'success': success,
                'ticket_id': ticket_id
            }
            
        else:
            return {
                'action': 'unknown',
                'success': False,
                'error': 'Could not determine the requested action'
            }
    
    def _determine_intent(self, user_input):
        """
        Use Bedrock to determine the user's intent from natural language.
        
        Args:
            user_input (str): User's natural language request
            
        Returns:
            dict: Intent information including action and relevant parameters
        """
        prompt = f"""
        You are a ticketing system agent. Analyze the following user request and extract the relevant information.
        Return a JSON object with the following structure:
        
        For ticket creation:
        {{
            "action": "create_ticket",
            "subject": "brief subject of the ticket",
            "description": "detailed description of the issue",
            "priority": "low|medium|high",
            "assigned_to": "person to assign the ticket to (optional)"
        }}
        
        For ticket status:
        {{
            "action": "get_ticket_status",
            "ticket_id": "the ID of the ticket"
        }}
        
        For ticket cancellation:
        {{
            "action": "cancel_ticket",
            "ticket_id": "the ID of the ticket",
            "reason": "reason for cancellation (optional)"
        }}
        
        User request: {user_input}
        
        JSON response:
        """
        
        # Call Bedrock model to process the prompt
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
        
        try:
            # Extract JSON from the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)
            return {"action": "unknown"}
        except (json.JSONDecodeError, ValueError):
            return {"action": "unknown"}