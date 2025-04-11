# TradeIQ Frontend API Integration Guide

This document provides guidance for frontend developers integrating with the TradeIQ assessment API.

## Base URL

All API requests should be made to:
```
http://localhost:8000/v1
```

## Authentication

The API uses Bearer token authentication:

```
Authorization: Bearer <token>
```

For testing, you can use:
```
Authorization: Bearer test-token
```

## Candlestick Pattern Assessment

### Endpoints

#### 1. Start Assessment

```
POST /candlestick-patterns/start
```

Start a new candlestick pattern assessment session.

**Request:**
```json
{
  "difficulty": 0.5,  // Optional: Default 0.5, range 0.0-1.0
  "total_questions": 5  // Optional: Default depends on server config
}
```

**Response:**
```json
{
  "question_id": "374f514f-04a0-47d0-93a8-3e29dcd2bc58",
  "question_number": 1,
  "total_questions": 5,
  "question_text": "What candlestick pattern is shown in this chart?",
  "options": [
    "Long-Legged Doji",
    "Dragonfly Doji",
    "Gravestone Doji",
    "Ladder Bottom"
  ],
  "image_data": "base64_encoded_image_placeholder",
  "time_limit_seconds": 18,
  "difficulty": 0.5,
  "session_id": "candlestick-822df652-0c88-4da6-b4ef-9465aa12faf9"
}
```

#### 2. Submit Answer

```
POST /candlestick-patterns/submit_answer
```

Submit an answer for evaluation and get the next question if available.

**Request:**
```json
{
  "session_id": "candlestick-822df652-0c88-4da6-b4ef-9465aa12faf9",
  "question_id": "374f514f-04a0-47d0-93a8-3e29dcd2bc58",
  "selected_option": "Gravestone Doji",
  "response_time_ms": 5000
}
```

**Response:**
```json
{
  "is_correct": true,
  "score": 10.0,
  "explanation": {
    "is_correct": true,
    "pattern_name": "Gravestone Doji",
    "user_level": "beginner",
    "components": {
      "pattern_definition": "...",
      "visual_characteristics": "...",
      "market_psychology": "...",
      "trading_implications": "...",
      "common_mistakes": "..."
    },
    "historical_examples": [
      "Example 1...",
      "Example 2..."
    ],
    "detection_details": {
      "confidence": 0.92,
      "detection_strategy": "weighted_consensus",
      "candle_indices": [3, 4, 5],
      "bullish": false,
      "contributing_detectors": [
        {"name": "GeometricPatternDetector", "confidence": 0.89, "weight": 1.2},
        {"name": "StatisticalPatternDetector", "confidence": 0.95, "weight": 0.8},
        {"name": "CNNPatternDetector", "confidence": 0.93, "weight": 1.0}
      ],
      "pattern_characteristics": {
        "trend_direction": "downtrend",
        "volume_confirmation": true,
        "key_levels": {
          "resistance": 155.67,
          "support": 152.43
        },
        "formation_quality": "high"
      },
      "alternative_interpretations": [
        {
          "pattern_name": "Evening Star",
          "confidence": 0.45,
          "explanation": "Some characteristics of this pattern resemble an Evening Star, but the middle candle body is too small."
        }
      ]
    }
  },
  "next_question": {
    // Same format as the response from /start
    // Will be null if assessment is complete
  },
  "assessment_complete": false,
  "accuracy": 0.75,
  "remaining_questions": 4
}
```

### Enhanced Pattern Detection Response

The `explanation.detection_details` field in the answer submission response provides rich information about how the pattern was detected:

1. **Core Detection Information**:
   - `confidence`: Overall confidence score (0.0-1.0) of the pattern detection
   - `detection_strategy`: The strategy used to detect the pattern (e.g., "rule_based", "ml_based", "weighted_consensus")
   - `candle_indices`: Array of indices identifying which candles form the pattern
   - `bullish`: Boolean indicating if the pattern is bullish (true), bearish (false), or neutral (null)

2. **Contributing Detectors**:
   - Details about individual detectors that contributed to the pattern identification
   - Each detector entry includes:
     - `name`: Detector class name
     - `confidence`: Individual confidence score from this detector
     - `weight`: Weight applied to this detector in the ensemble

3. **Pattern Characteristics**:
   - `trend_direction`: The market trend context for the pattern
   - `volume_confirmation`: Whether volume data supports the pattern significance
   - `key_levels`: Important price levels related to the pattern
   - `formation_quality`: Qualitative assessment of pattern quality

4. **Alternative Interpretations**:
   - Other patterns that could potentially match but with lower confidence
   - Useful for understanding pattern ambiguity and related formations

This enhanced detection information can be used to:
- Display detailed explanations to advanced users
- Provide context for why a pattern was identified
- Help users understand pattern ambiguity and related formations
- Visualize the specific candles that form the pattern

#### 3. Get Session Details

```
GET /candlestick-patterns/session/{session_id}
```

Retrieve details about an ongoing or completed session.

**Response:**
```json
{
  "session_id": "candlestick-822df652-0c88-4da6-b4ef-9465aa12faf9",
  "total_questions": 5,
  "completed_questions": 2,
  "correct_answers": 1,
  "avg_response_time": 3500,
  "score": 10.0,
  "started_at": 1648723456,
  "completed_at": null,  // Will be populated when all questions are answered
  "user_id": "user123",
  "accuracy": 0.5
}
```

#### 4. Get User History

```
GET /candlestick-patterns/history?user_id={user_id}
```

Retrieve history of assessment sessions for a user.

**Response:**
```json
{
  "sessions": [
    {
      "session_id": "candlestick-822df652-0c88-4da6-b4ef-9465aa12faf9",
      "started_at": 1648723456,
      "completed_at": 1648723756,
      "total_questions": 5,
      "completed_questions": 5,
      "correct_answers": 3,
      "score": 30.0,
      "accuracy": 0.6
    },
    // More sessions...
  ],
  "total_sessions": 1,
  "total_correct_answers": 3,
  "total_questions_answered": 5,
  "average_accuracy": 0.6
}
```

## Error Handling

The API returns standard HTTP status codes:

- `200 OK` - Request succeeded
- `400 Bad Request` - Invalid request format
- `401 Unauthorized` - Authentication required
- `404 Not Found` - Resource not found
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

Error responses have the following format:

```json
{
  "detail": "Error message explaining what went wrong"
}
```

## Implementation Tips

1. **Session Management**:
   - Store the `session_id` received from the `/start` endpoint
   - Use this `session_id` for all subsequent requests

2. **Timer Implementation**:
   - Use the `time_limit_seconds` from the question to implement a countdown timer
   - Calculate `response_time_ms` as the time difference between question display and answer submission

3. **Image Handling**:
   - The `image_data` field contains a Base64-encoded image
   - Decode and display this image with the question

4. **Error Handling**:
   - Implement robust error handling for network issues
   - Add retry logic for transient failures
   - Display user-friendly error messages

5. **Session Resumption**:
   - If a user refreshes the page or returns later, use the `GET /session/{session_id}` endpoint to resume

## Testing

For testing the integration:

1. Start a new session with `POST /candlestick-patterns/start`
2. Submit answers with `POST /candlestick-patterns/submit_answer`
3. Check session details with `GET /candlestick-patterns/session/{session_id}`
4. View history with `GET /candlestick-patterns/history?user_id={user_id}`

Use tools like Postman or curl to test individual endpoints before integrating them into your frontend code. 

## Handling Completed Assessments

When an assessment is complete, the response from `POST /candlestick-patterns/submit_answer` endpoint will have some special considerations:

1. **Assessment Complete Flag**:
   - `assessment_complete` will be `true`
   - This indicates that the user has finished all questions in the assessment

2. **Response Structure for Completed Assessments**:
   ```json
   {
     "is_correct": true,
     "score": 10.0,
     "explanation": {
       "is_correct": true,
       "pattern_name": "Gravestone Doji",
       "user_level": "beginner",
       "components": {
         "pattern_definition": "...",
         "visual_characteristics": "...",
         "market_psychology": "...",
         "trading_implications": "...",
         "common_mistakes": "..."
       },
       "historical_examples": [
         "Example 1...",
         "Example 2..."
       ]
     },
     "assessment_complete": true,
     "accuracy": 0.75,
     "remaining_questions": 0
   }
   ```

3. **Key Differences from Ongoing Assessment Responses**:
   - The `next_question` field is not included in completed assessment responses
   - `remaining_questions` will be `0`
   - `assessment_complete` will be `true`

4. **Frontend Implementation**:
   - Check for `assessment_complete: true` before attempting to display the next question
   - When an assessment is complete, redirect to a summary page or display completion information
   - Use the `GET /candlestick-patterns/session/{session_id}` endpoint to retrieve complete session statistics

5. **Submitting Answers after Completion**:
   - The API will respond with an appropriate error status if you attempt to submit answers after an assessment is complete
   - Your frontend should prevent users from submitting additional answers after completion 