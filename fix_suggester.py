"""
Fix Suggester Module - Complete Working Version
Suggests fixes based on issue type and keywords
"""

import re

class FixSuggester:
    """Suggest fixes for bugs and features"""
    
    def __init__(self):
        self.fix_database = {
            # Null pointer exceptions
            'nullpointer|null pointer': [
                "Add null check before accessing object methods: if (object != null) { object.method(); }",
                "Initialize object before use: Object obj = new Object();",
                "Use Optional class: Optional.ofNullable(object).ifPresent(obj -> obj.method())",
                "Add @Nullable and @NotNull annotations for better null safety"
            ],
            
            # SQL injection
            'sql injection|sql injection': [
                "Use parameterized queries: PreparedStatement stmt = conn.prepareStatement('SELECT * FROM users WHERE username = ?')",
                "Implement input validation and sanitization for all user inputs",
                "Use an ORM framework like Hibernate with built-in SQL injection protection",
                "Apply the principle of least privilege for database users"
            ],
            
            # Deadlock
            'deadlock|dead lock': [
                "Ensure consistent lock ordering across all transactions",
                "Reduce transaction scope and duration to minimize lock contention",
                "Use optimistic locking instead of pessimistic locking",
                "Implement retry logic with exponential backoff"
            ],
            
            # Memory leak
            'memory leak|outofmemory|oom': [
                "Close resources (files, connections) in finally block: try { } finally { resource.close(); }",
                "Use try-with-resources in Java: try (FileInputStream fis = new FileInputStream(file))",
                "Implement proper garbage collection: System.gc() (use carefully)",
                "Check for event listener leaks and remove them when not needed"
            ],
            
            # Timeout
            'timeout|timed out': [
                "Optimize database queries with proper indexes",
                "Implement connection pooling with appropriate pool sizes",
                "Increase timeout configuration: server.servlet.session.timeout=30m",
                "Add retry logic with exponential backoff for external calls"
            ],
            
            # Authentication
            'authentication|auth|login': [
                "Validate all authentication inputs server-side",
                "Implement rate limiting: 5 attempts per minute",
                "Use secure session management with HTTP-only cookies",
                "Add CSRF protection: @EnableWebSecurity in Spring Security"
            ],
            
            # API
            'api|endpoint|rest': [
                "Add proper error handling with try-catch blocks",
                "Implement request validation using @Valid annotation",
                "Add comprehensive logging: log.error('Error: ', exception)",
                "Use proper HTTP status codes: 400 for bad request, 500 for server error"
            ],
            
            # File upload
            'upload|file upload': [
                "Increase file size limit: spring.servlet.multipart.max-file-size=10MB",
                "Validate file types and sizes before processing",
                "Store files securely outside the web root",
                "Implement chunked upload for large files"
            ],
            
            # Database
            'database|db|sql': [
                "Add database indexes on frequently queried columns",
                "Optimize queries with EXPLAIN plan analysis",
                "Use connection pooling: HikariCP with proper configuration",
                "Implement query caching for frequently accessed data"
            ],
            
            # UI/UX
            'ui|ux|interface|layout': [
                "Implement responsive design with CSS media queries",
                "Use virtualization for large lists: react-window or react-virtualized",
                "Add loading states and error boundaries",
                "Optimize images and assets for faster loading"
            ],
            
            # Performance
            'performance|slow|fast': [
                "Implement caching with Redis or Memcached",
                "Use lazy loading for expensive operations",
                "Optimize database queries with proper indexes",
                "Add pagination for large datasets"
            ],
            
            # Security
            'security|vulnerability|exploit': [
                "Keep all dependencies updated to latest versions",
                "Implement HTTPS with valid SSL certificates",
                "Use security headers: CSP, X-Frame-Options, HSTS",
                "Regular security audits and penetration testing"
            ]
        }
        
        self.feature_suggestions = {
            'oauth|google|facebook|login': [
                "Implement OAuth2.0 with Spring Security OAuth",
                "Use established libraries like Spring Security or Passport.js",
                "Store refresh tokens securely in database",
                "Implement token rotation for better security"
            ],
            
            'dark mode|theme': [
                "Use CSS variables for theme colors: :root { --bg-color: white; }",
                "Implement with React Context or Redux for state management",
                "Store preference in localStorage: localStorage.setItem('theme', 'dark')",
                "Detect system preference: @media (prefers-color-scheme: dark)"
            ],
            
            'export|pdf|csv|excel': [
                "Use libraries like iText (PDF) or Apache POI (Excel)",
                "Implement async processing for large exports",
                "Add progress indicator for better user feedback",
                "Generate files in background and email when ready"
            ],
            
            'dashboard|analytics|chart': [
                "Use charting libraries like Chart.js, D3.js, or Recharts",
                "Implement real-time updates with WebSockets",
                "Add filter and drill-down capabilities",
                "Cache dashboard data for better performance"
            ],
            
            '2fa|two factor|mfa': [
                "Implement with Google Authenticator using TOTP algorithm",
                "Store backup codes encrypted in database",
                "Add SMS fallback option using Twilio API",
                "Implement remember device for 30 days"
            ],
            
            'search|filter': [
                "Use Elasticsearch for full-text search capabilities",
                "Implement fuzzy search for handling typos: 'John~2'",
                "Add faceted filters for better UX",
                "Cache search results for frequently used queries"
            ],
            
            'notification|alert|email': [
                "Use message queues like RabbitMQ for async processing",
                "Implement retry logic with exponential backoff",
                "Add unsubscribe functionality for email notifications",
                "Use templates for consistent formatting"
            ],
            
            'pwa|progressive web app': [
                "Add service worker for offline support",
                "Generate manifest.json with app icons",
                "Implement push notifications",
                "Add to home screen prompt"
            ],
            
            'api|rest|graphql': [
                "Document API using Swagger/OpenAPI",
                "Implement rate limiting: 100 requests per minute",
                "Add pagination: ?page=1&limit=20",
                "Use API versioning: /api/v1/, /api/v2/"
            ],
            
            'testing|test|unit test': [
                "Write unit tests with JUnit or pytest",
                "Add integration tests for API endpoints",
                "Implement CI/CD pipeline with GitHub Actions",
                "Achieve 80% code coverage minimum"
            ]
        }
    
    def suggest_fix(self, text, prediction_type):
        """Suggest fixes based on issue text"""
        if not isinstance(text, str):
            text = str(text) if text else ""
        
        text_lower = text.lower()
        suggestions = []
        
        if prediction_type == "Bug":
            # Check each bug pattern
            for pattern, fixes in self.fix_database.items():
                if any(keyword in text_lower for keyword in pattern.split('|')):
                    suggestions.extend(fixes)
                    if len(suggestions) >= 3:
                        break
            
            # Generic bug fixes if no specific pattern found
            if not suggestions:
                suggestions = [
                    "Add comprehensive error handling with try-catch blocks",
                    "Implement proper logging: log.error('Error details: ', exception)",
                    "Add unit tests to reproduce and verify the issue",
                    "Review recent code changes in affected module",
                    "Check stack trace for exact line numbers and fix accordingly"
                ]
        else:
            # Feature suggestions
            for pattern, suggestions_list in self.feature_suggestions.items():
                if any(keyword in text_lower for keyword in pattern.split('|')):
                    suggestions.extend(suggestions_list)
                    if len(suggestions) >= 3:
                        break
            
            # Generic feature suggestions
            if not suggestions:
                suggestions = [
                    "Create user stories and acceptance criteria",
                    "Design UI/UX mockups first using Figma or Sketch",
                    "Break down into smaller, manageable tasks",
                    "Consider performance and scalability implications",
                    "Add documentation and update API specs"
                ]
        
        # Return top 3 unique suggestions
        unique_suggestions = []
        for s in suggestions:
            if s not in unique_suggestions:
                unique_suggestions.append(s)
        
        return unique_suggestions[:3]
    
    def get_detailed_fix(self, text, prediction_type):
        """Get more detailed fix with code examples"""
        suggestions = self.suggest_fix(text, prediction_type)
        
        detailed = []
        for suggestion in suggestions:
            if 'null' in text.lower() or 'nullpointer' in text.lower():
                detailed.append({
                    'title': 'Null Check Implementation',
                    'description': suggestion,
                    'code': '''
// Add null check before accessing object
if (object != null) {
    object.method();
} else {
    logger.error("Object is null at " + new Exception().getStackTrace()[0]);
    // Handle null case appropriately
    return default value or throw custom exception
}
'''
                })
            elif 'sql' in text.lower() or 'injection' in text.lower():
                detailed.append({
                    'title': 'Parameterized Query',
                    'description': suggestion,
                    'code': '''
// Use parameterized queries to prevent SQL injection
String sql = "SELECT * FROM users WHERE username = ? AND password = ?";
PreparedStatement pstmt = connection.prepareStatement(sql);
pstmt.setString(1, username);
pstmt.setString(2, hashedPassword);
ResultSet rs = pstmt.executeQuery();
'''
                })
            elif 'timeout' in text.lower():
                detailed.append({
                    'title': 'Timeout Configuration',
                    'description': suggestion,
                    'code': '''
// Configure timeout in application.properties
# Connection timeout
spring.datasource.hikari.connection-timeout=30000
# Socket timeout
spring.datasource.hikari.socket-timeout=60000
# Transaction timeout
spring.transaction.default-timeout=30
'''
                })
            else:
                detailed.append({
                    'title': 'Suggested Fix',
                    'description': suggestion,
                    'code': None
                })
        
        return detailed