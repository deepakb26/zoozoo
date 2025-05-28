import os
import json
import argparse
from secure_multi_agent_system import SecureMultiAgentSystem

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Multi-Agent Collaboration System')
    
    parser.add_argument('--supervisor-agent-id', 
                        help='Bedrock agent ID for the supervisor')
    parser.add_argument('--supervisor-agent-alias-id', 
                        help='Bedrock agent alias ID for the supervisor')
    parser.add_argument('--model-id', 
                        default='anthropic.claude-3-sonnet-20240229-v1:0',
                        help='Bedrock model ID to use for sub-agents')
    parser.add_argument('--region-name', 
                        default='us-east-1',
                        help='AWS region name')
    parser.add_argument('--s3-bucket', 
                        help='S3 bucket containing the knowledge base')
    parser.add_argument('--sns-topic-arn', 
                        help='ARN of the SNS topic for emergency notifications')
    parser.add_argument('--guardrail-id', 
                        help='ID of the Bedrock guardrail')
    parser.add_argument('--guardrail-version', 
                        help='Version of the Bedrock guardrail')
    parser.add_argument('--interactive', 
                        action='store_true',
                        help='Run in interactive mode')
    
    return parser.parse_args()

def main():
    """Main entry point for the multi-agent system."""
    args = parse_arguments()
    
    # Use environment variables as fallback
    supervisor_agent_id = args.supervisor_agent_id or os.environ.get('SUPERVISOR_AGENT_ID')
    supervisor_agent_alias_id = args.supervisor_agent_alias_id or os.environ.get('SUPERVISOR_AGENT_ALIAS_ID')
    model_id = args.model_id or os.environ.get('BEDROCK_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')
    region_name = args.region_name or os.environ.get('AWS_REGION', 'us-east-1')
    s3_bucket = args.s3_bucket or os.environ.get('KNOWLEDGE_BASE_S3_BUCKET')
    sns_topic_arn = args.sns_topic_arn or os.environ.get('EMERGENCY_SNS_TOPIC_ARN')
    guardrail_id = args.guardrail_id or os.environ.get('BEDROCK_GUARDRAIL_ID')
    guardrail_version = args.guardrail_version or os.environ.get('BEDROCK_GUARDRAIL_VERSION')
    
    # Initialize the secure multi-agent system
    system = SecureMultiAgentSystem(
        supervisor_agent_id=supervisor_agent_id,
        supervisor_agent_alias_id=supervisor_agent_alias_id,
        model_id=model_id,
        region_name=region_name,
        s3_bucket=s3_bucket,
        sns_topic_arn=sns_topic_arn,
        guardrail_id=guardrail_id,
        guardrail_version=guardrail_version
    )
    
    if args.interactive:
        # Interactive mode
        session_id = None
        print("Multi-Agent Collaboration System")
        print("Type 'exit' to quit")
        print("-" * 50)
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            response = system.process_request(user_input, session_id)
            
            # Update session ID if available
            if isinstance(response, dict) and 'session_id' in response:
                session_id = response.get('session_id')
            
            # Pretty print the response
            if isinstance(response, dict):
                # Extract the main message for display
                if 'message' in response:
                    print(f"\nSystem: {response['message']}")
                elif 'answer' in response:
                    print(f"\nSystem: {response['answer']}")
                elif 'response' in response:
                    print(f"\nSystem: {response['response']}")
                else:
                    print(f"\nSystem: {json.dumps(response, indent=2)}")
                
                # Show additional information if available
                if 'actions' in response and response['actions']:
                    print("\nRecommended actions:")
                    for i, action in enumerate(response['actions'], 1):
                        print(f"  {i}. {action}")
                
                if 'sources' in response and response['sources']:
                    print("\nSources:")
                    for source in response['sources']:
                        print(f"  - {source}")
            else:
                print(f"\nSystem: {response}")
    else:
        # Non-interactive mode (for use in Lambda or other services)
        print("Multi-Agent Collaboration System initialized")
        print(f"Supervisor Agent ID: {supervisor_agent_id}")
        print(f"Model ID: {model_id}")
        print(f"Region: {region_name}")
        print(f"Guardrails enabled: {guardrail_id is not None}")

if __name__ == "__main__":
    main()