import unittest
from unittest.mock import patch, MagicMock
from ticketing_agent import TicketingAgent
import uuid

class TestTicketingAgent(unittest.TestCase):
    """Test cases for the TicketingAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_table = MagicMock()
        with patch('boto3.resource') as mock_resource:
            mock_dynamodb = MagicMock()
            mock_resource.return_value = mock_dynamodb
            mock_dynamodb.Table.return_value = self.mock_table
            self.agent = TicketingAgent()
    
    def test_create_ticket(self):
        """Test creating a ticket."""
        # Mock UUID to get a consistent ticket ID
        test_uuid = "test-uuid-1234"
        with patch('uuid.uuid4', return_value=uuid.UUID(test_uuid)):
            ticket = self.agent.create_ticket(
                subject="Test Ticket",
                description="This is a test ticket",
                priority="high"
            )
            
            # Verify the ticket was created with correct data
            self.assertEqual(ticket["ticket_id"], test_uuid)
            self.assertEqual(ticket["subject"], "Test Ticket")
            self.assertEqual(ticket["description"], "This is a test ticket")
            self.assertEqual(ticket["priority"], "high")
            self.assertEqual(ticket["status"], "open")
            
            # Verify put_item was called
            self.mock_table.put_item.assert_called_once()
    
    def test_get_ticket_status_found(self):
        """Test getting a ticket that exists."""
        # Mock the DynamoDB response
        self.mock_table.get_item.return_value = {
            'Item': {
                'ticket_id': 'test-123',
                'subject': 'Test Ticket',
                'status': 'open'
            }
        }
        
        ticket = self.agent.get_ticket_status('test-123')
        
        # Verify the ticket was returned
        self.assertIsNotNone(ticket)
        self.assertEqual(ticket['ticket_id'], 'test-123')
        self.assertEqual(ticket['status'], 'open')
        
        # Verify get_item was called with correct key
        self.mock_table.get_item.assert_called_with(Key={'ticket_id': 'test-123'})
    
    def test_get_ticket_status_not_found(self):
        """Test getting a ticket that doesn't exist."""
        # Mock the DynamoDB response for a non-existent ticket
        self.mock_table.get_item.return_value = {}
        
        ticket = self.agent.get_ticket_status('nonexistent-123')
        
        # Verify None was returned
        self.assertIsNone(ticket)
    
    def test_cancel_ticket(self):
        """Test cancelling a ticket."""
        # Mock the DynamoDB response
        self.mock_table.update_item.return_value = {
            'Attributes': {
                'status': 'cancelled'
            }
        }
        
        result = self.agent.cancel_ticket('test-123', reason='No longer needed')
        
        # Verify the result
        self.assertTrue(result)
        
        # Verify update_item was called with correct parameters
        self.mock_table.update_item.assert_called_once()
        args, kwargs = self.mock_table.update_item.call_args
        self.assertEqual(kwargs['Key'], {'ticket_id': 'test-123'})
        self.assertIn(':status', kwargs['ExpressionAttributeValues'])
        self.assertEqual(kwargs['ExpressionAttributeValues'][':status'], 'cancelled')
        self.assertIn(':reason', kwargs['ExpressionAttributeValues'])
        self.assertEqual(kwargs['ExpressionAttributeValues'][':reason'], 'No longer needed')

if __name__ == '__main__':
    unittest.main()