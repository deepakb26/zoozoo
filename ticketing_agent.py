import boto3
import uuid
from datetime import datetime
from botocore.exceptions import ClientError

class TicketingAgent:
    """
    A ticketing agent that interacts with DynamoDB to manage tickets.
    """
    
    def __init__(self, table_name="Ticket-DB", region_name="us-east-1"):
        """
        Initialize the ticketing agent with the DynamoDB table.
        
        Args:
            table_name (str): Name of the DynamoDB table
            region_name (str): AWS region name
        """
        self.dynamodb = boto3.resource('dynamodb', region_name=region_name)
        self.table = self.dynamodb.Table(table_name)
    
    def create_ticket(self, subject, description, user_id,emergency, priority="medium", assigned_to=None):
        """
        Create a new ticket in the DynamoDB table.
        
        Args:
            subject (str): Subject of the ticket
            description (str): Detailed description of the ticket
            priority (str): Priority level (low, medium, high)
            assigned_to (str, optional): Person assigned to the ticket
            
        Returns:
            dict: The created ticket information
        """
        ticket_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        ticket = {
            'ticket_id': ticket_id,
            'subject': subject,
            'user_id':user_id,
            'assigned_to': assigned_to,
            'description': description,
            'status': 'open',
            'priority': priority,
            'created_at': timestamp,
            'updated_at': timestamp,
            'emergency':emergency
        }
        
        if assigned_to:
            ticket['assigned_to'] = assigned_to
        
        try:
            self.table.put_item(Item=ticket)
            return ticket
        except ClientError as e:
            print(f"Error creating ticket: {e}")
            return None
    
    def get_ticket_status(self, ticket_id):
        """
        Get the status and details of a ticket.
        
        Args:
            ticket_id (str): ID of the ticket to retrieve
            
        Returns:
            dict: The ticket information or None if not found
        """
        try:
            response = self.table.get_item(Key={'ticket_id': ticket_id})
            if 'Item' in response:
                return response['Item']
            else:
                print(f"Ticket {ticket_id} not found")
                return None
        except ClientError as e:
            print(f"Error retrieving ticket: {e}")
            return None
    
    def cancel_ticket(self, ticket_id, reason=None):
        """
        Cancel a ticket by updating its status to 'cancelled'.
        
        Args:
            ticket_id (str): ID of the ticket to cancel
            reason (str, optional): Reason for cancellation
            
        Returns:
            bool: True if successful, False otherwise
        """
        timestamp = datetime.utcnow().isoformat()
        
        update_expression = "SET #status = :status, updated_at = :updated"
        expression_values = {
            ':status': 'cancelled',
            ':updated': timestamp
        }
        
        if reason:
            update_expression += ", cancellation_reason = :reason"
            expression_values[':reason'] = reason
        
        try:
            self.table.update_item(
                Key={'ticket_id': ticket_id},
                UpdateExpression=update_expression,
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues=expression_values,
                ReturnValues="UPDATED_NEW"
            )
            return True
        except ClientError as e:
            print(f"Error cancelling ticket: {e}")
            return False