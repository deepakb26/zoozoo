import os
import json
from secure_multi_agent_system import SecureMultiAgentSystem

def lambda_handler(event, context):
    """
    AWS Lambda handler function for the multi-agent system.
    
    Args:
        event (dict): Lambda event data
        context (object): Lambda context
        
    Returns:
        dict: Response containing the result of the operation
    """
    # Extract user input from the event
    body = json.loads(event.get('body', '{}'))
    user_input = body.get('input', '')
    session_id = body.get('session_id')
    
    if not user_input:
        return {
            'statusCode': 400,
            'body': json.dumps({
                'error': 'No input provided'
            })
        }
    
    # Initialize the secure multi-agent system
    system = SecureMultiAgentSystem(
        supervisor_agent_id=os.environ.get('SUPERVISOR_AGENT_ID'),
        supervisor_agent_alias_id=os.environ.get('SUPERVISOR_AGENT_ALIAS_ID'),
        model_id=os.environ.get('BEDROCK_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0'),
        region_name=os.environ.get('AWS_REGION', 'us-east-1'),
        s3_bucket=os.environ.get('KNOWLEDGE_BASE_S3_BUCKET'),
        sns_topic_arn=os.environ.get('EMERGENCY_SNS_TOPIC_ARN'),
        guardrail_id=os.environ.get('BEDROCK_GUARDRAIL_ID'),
        guardrail_version=os.environ.get('BEDROCK_GUARDRAIL_VERSION')
    )
    
    # Process the request
    result = system.process_request(user_input, session_id)
    
    # Extract session ID if available
    response_body = result
    if isinstance(result, dict) and 'session_id' in result:
        session_id = result.pop('session_id', None)
        response_body = {
            'response': result,
            'session_id': session_id
        }
    
    return {
        'statusCode': 200,
        'body': json.dumps(response_body)
    }