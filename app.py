import os
import json
from bedrock_integration import BedrockTicketingAgent

def lambda_handler(event, context):
    """
    AWS Lambda handler function for processing ticketing requests.
    
    Args:
        event (dict): Lambda event data
        context (object): Lambda context
        
    Returns:
        dict: Response containing the result of the ticketing operation
    """
    # Extract user input from the event
    body = json.loads(event.get('body', '{}'))
    user_input = body.get('input', '')
    
    if not user_input:
        return {
            'statusCode': 400,
            'body': json.dumps({
                'error': 'No input provided'
            })
        }
    
    # Initialize the Bedrock ticketing agent
    model_id = os.environ.get('BEDROCK_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')
    region_name = os.environ.get('AWS_REGION', 'us-east-1')
    
    agent = BedrockTicketingAgent(model_id=model_id, region_name=region_name)
    
    # Process the request
    result = agent.process_request(user_input)
    
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }

if __name__ == "__main__":
    # For local testing
    test_input = input("Enter your ticketing request: ")
    agent = BedrockTicketingAgent()
    result = agent.process_request(test_input)
    print(json.dumps(result, indent=2))