"""
Fix Suggester Module - Suggests fixes based on issue type
"""

class FixSuggester:
    """Suggest fixes for bugs and features"""
    
    def __init__(self):
        pass
    
    def suggest_fix(self, text, prediction_type):
        """Suggest fixes based on issue text"""
        if not isinstance(text, str):
            text = str(text) if text else ""
        
        text_lower = text.lower()
        suggestions = []
        
        if prediction_type == "Bug":
            # Bug fixes
            if 'null' in text_lower or 'nullpointer' in text_lower:
                suggestions = [
                    "Add null check before accessing object",
                    "Initialize object before use",
                    "Use Optional class to handle null values"
                ]
            elif 'sql' in text_lower or 'injection' in text_lower:
                suggestions = [
                    "Use parameterized queries",
                    "Implement input validation",
                    "Use ORM with built-in SQL injection protection"
                ]
            elif 'deadlock' in text_lower:
                suggestions = [
                    "Ensure consistent lock ordering",
                    "Reduce transaction scope",
                    "Use optimistic locking instead of pessimistic"
                ]
            elif 'memory' in text_lower or 'leak' in text_lower:
                suggestions = [
                    "Close resources in finally block",
                    "Use try-with-resources",
                    "Implement proper garbage collection"
                ]
            elif 'timeout' in text_lower:
                suggestions = [
                    "Optimize database queries with indexes",
                    "Implement connection pooling",
                    "Increase timeout configuration"
                ]
            elif 'authentication' in text_lower or 'login' in text_lower:
                suggestions = [
                    "Validate all inputs server-side",
                    "Implement rate limiting",
                    "Use secure session management"
                ]
            elif 'api' in text_lower or 'endpoint' in text_lower:
                suggestions = [
                    "Add proper error handling",
                    "Implement request validation",
                    "Add comprehensive logging"
                ]
            else:
                suggestions = [
                    "Add comprehensive error handling",
                    "Implement proper logging",
                    "Add unit tests to reproduce the issue"
                ]
        else:
            # Feature suggestions
            if 'oauth' in text_lower or 'login' in text_lower or 'authentication' in text_lower:
                suggestions = [
                    "Implement OAuth2.0 with Spring Security",
                    "Use established libraries like Spring Security OAuth",
                    "Store tokens securely in database"
                ]
            elif 'dark' in text_lower or 'theme' in text_lower or 'mode' in text_lower:
                suggestions = [
                    "Use CSS variables for theme colors",
                    "Implement with React Context",
                    "Store preference in localStorage"
                ]
            elif 'pdf' in text_lower or 'export' in text_lower or 'csv' in text_lower:
                suggestions = [
                    "Use libraries like iText for PDF",
                    "Implement async processing for large exports",
                    "Add progress indicator for user feedback"
                ]
            elif 'dashboard' in text_lower or 'analytics' in text_lower or 'chart' in text_lower:
                suggestions = [
                    "Use charting libraries like Chart.js",
                    "Implement real-time updates with WebSockets",
                    "Add filter and drill-down capabilities"
                ]
            elif '2fa' in text_lower or 'two factor' in text_lower or 'mfa' in text_lower:
                suggestions = [
                    "Implement with Google Authenticator TOTP",
                    "Store backup codes encrypted",
                    "Add SMS fallback option"
                ]
            elif 'search' in text_lower:
                suggestions = [
                    "Use Elasticsearch for full-text search",
                    "Implement fuzzy search for typos",
                    "Add faceted filters for better UX"
                ]
            else:
                suggestions = [
                    "Create user stories and acceptance criteria",
                    "Design UI/UX mockups first",
                    "Break down into smaller tasks"
                ]
        
        return suggestions[:3]  # Return top 3 suggestions