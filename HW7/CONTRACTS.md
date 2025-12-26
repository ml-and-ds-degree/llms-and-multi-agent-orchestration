# Protocol Contracts Reference

**Protocol Version**: `league.v2`

---

## 1. LEAGUE_MANAGER SOURCE

### 1.1 ROUND_ANNOUNCEMENT
**Direction**: League Manager → All Agents  
**When**: New round begins

**Fields:**

| Field | Type | Required |
|-------|------|----------|
| `message_type` | string | ✅ |
| `league_id` | string | ✅ |
| `round_id` | integer | ✅ |
| `matches` | array | ✅ |

**Example:**
```json
{
  "protocol": "league.v2",
  "message_type": "ROUND_ANNOUNCEMENT",
  "sender": "league_manager",
  "timestamp": "2025-01-15T10:00:00Z",
  "conversation_id": "conv-round1-announcement",
  "league_id": "league_2025_even_odd",
  "round_id": 1,
  "matches": [
    {
      "match_id": "R1M1",
      "game_type": "even_odd",
      "player_A_id": "P01",
      "player_B_id": "P02",
      "referee_endpoint": "http://localhost:8001/mcp"
    },
    {
      "match_id": "R1M2",
      "game_type": "even_odd",
      "player_A_id": "P03",
      "player_B_id": "P04",
      "referee_endpoint": "http://localhost:8002/mcp"
    }
  ]
}
```

---

### 1.2 ROUND_COMPLETED
**Direction**: League Manager → Players  
**When**: All matches in round completed

**Fields:**

| Field | Type | Required |
|-------|------|----------|
| `protocol` | string | ✅ |
| `message_type` | string | ✅ |
| `sender` | string | ✅ |
| `timestamp` | ISO-8601 | ✅ |
| `conversation_id` | string | ✅ |
| `league_id` | string | ✅ |
| `round_id` | integer | ✅ |
| `matches_completed` | integer | ✅ |
| `next_round_id` | integer | ❌ |
| `summary` | object | ✅ |

**Example:**
```json
{
  "protocol": "league.v2",
  "message_type": "ROUND_COMPLETED",
  "sender": "league_manager",
  "timestamp": "2025-01-15T12:00:00Z",
  "conversation_id": "conv-round1-complete",
  "league_id": "league_2025_even_odd",
  "round_id": 1,
  "matches_completed": 2,
  "next_round_id": 2,
  "summary": {
    "total_matches": 2,
    "wins": 1,
    "draws": 1,
    "technical_losses": 0
  }
}
```

---

### 1.3 LEAGUE_STANDINGS_UPDATE
**Direction**: League Manager → Players  
**When**: After each round completion

**Fields:**

| Field | Type | Required |
|-------|------|----------|
| `message_type` | string | ✅ |
| `league_id` | string | ✅ |
| `round_id` | integer | ✅ |
| `standings` | array | ✅ |

**Example:**
```json
{
  "protocol": "league.v2",
  "message_type": "LEAGUE_STANDINGS_UPDATE",
  "sender": "league_manager",
  "timestamp": "2025-01-15T12:05:00Z",
  "conversation_id": "conv-standings-update-r1",
  "league_id": "league_2025_even_odd",
  "round_id": 1,
  "standings": [
    {
      "rank": 1,
      "player_id": "P01",
      "display_name": "Agent Alpha",
      "played": 2,
      "wins": 2,
      "draws": 0,
      "losses": 0,
      "points": 6
    },
    {
      "rank": 2,
      "player_id": "P03",
      "display_name": "Agent Gamma",
      "played": 2,
      "wins": 1,
      "draws": 1,
      "losses": 0,
      "points": 4
    }
  ]
}
```

---

### 1.4 LEAGUE_COMPLETED
**Direction**: League Manager → All Agents  
**When**: All rounds completed

**Fields:**

| Field | Type | Required |
|-------|------|----------|
| `protocol` | string | ✅ |
| `message_type` | string | ✅ |
| `sender` | string | ✅ |
| `timestamp` | ISO-8601 | ✅ |
| `conversation_id` | string | ✅ |
| `league_id` | string | ✅ |
| `total_rounds` | integer | ✅ |
| `total_matches` | integer | ✅ |
| `champion` | object | ✅ |
| `final_standings` | array | ✅ |

**Example:**
```json
{
  "protocol": "league.v2",
  "message_type": "LEAGUE_COMPLETED",
  "sender": "league_manager",
  "timestamp": "2025-01-20T18:00:00Z",
  "conversation_id": "conv-league-complete",
  "league_id": "league_2025_even_odd",
  "total_rounds": 3,
  "total_matches": 6,
  "champion": {
    "player_id": "P01",
    "display_name": "Agent Alpha",
    "points": 9
  },
  "final_standings": [
    {"rank": 1, "player_id": "P01", "points": 9},
    {"rank": 2, "player_id": "P03", "points": 5},
    {"rank": 3, "player_id": "P02", "points": 3},
    {"rank": 4, "player_id": "P04", "points": 1}
  ]
}
```

---

### 1.5 LEAGUE_QUERY_RESPONSE
**Direction**: League Manager → Player/Referee  
**When**: Response to LEAGUE_QUERY

**Fields:**

| Field | Type | Required |
|-------|------|----------|
| `protocol` | string | ✅ |
| `message_type` | string | ✅ |
| `sender` | string | ✅ |
| `timestamp` | ISO-8601 | ✅ |
| `conversation_id` | string | ✅ |
| `query_type` | string | ✅ |
| `success` | boolean | ✅ |
| `data` | object | ❌ |

**Example:**
```json
{
  "protocol": "league.v2",
  "message_type": "LEAGUE_QUERY_RESPONSE",
  "sender": "league_manager",
  "timestamp": "2025-01-15T14:00:01Z",
  "conversation_id": "conv-query-001",
  "query_type": "GET_NEXT_MATCH",
  "success": true,
  "data": {
    "next_match": {
      "match_id": "R2M1",
      "round_id": 2,
      "opponent_id": "P03",
      "referee_endpoint": "http://localhost:8001/mcp"
    }
  }
}
```

---

### 1.6 LEAGUE_ERROR
**Direction**: League Manager → Agent  
**When**: League-level error occurs

**Fields:**

| Field | Type | Required |
|-------|------|----------|
| `protocol` | string | ✅ |
| `message_type` | string | ✅ |
| `sender` | string | ✅ |
| `timestamp` | ISO-8601 | ✅ |
| `conversation_id` | string | ✅ |
| `error_code` | string | ✅ |
| `error_description` | string | ✅ |
| `original_message_type` | string | ❌ |
| `context` | object | ❌ |

**Example:**
```json
{
  "protocol": "league.v2",
  "message_type": "LEAGUE_ERROR",
  "sender": "league_manager",
  "timestamp": "2025-01-15T10:05:30Z",
  "conversation_id": "conv-error-001",
  "error_code": "E012",
  "error_description": "AUTH_TOKEN_INVALID",
  "original_message_type": "LEAGUE_QUERY",
  "context": {
    "provided_token": "tok-invalid-xxx",
    "expected_format": "tok-{agent_id}-{hash}"
  }
}
```

---

## 2. REFEREE SOURCE

### 2.1 REFEREE_REGISTER_REQUEST
**Direction**: Referee → League Manager  
**Endpoint**: `POST /mcp`  
**When**: Referee starts up

**Fields:**

| Field | Type | Required |
|-------|------|----------|
| `message_type` | string | ✅ |
| `referee_meta` | object | ✅ |

**Example:**
```json
{
  "protocol": "league.v2",
  "message_type": "REFEREE_REGISTER_REQUEST",
  "sender": "referee:REF01",
  "timestamp": "2025-01-15T09:00:00Z",
  "conversation_id": "conv-ref01-registration",
  "referee_meta": {
    "display_name": "Referee Alpha",
    "version": "1.0.0",
    "game_types": ["even_odd"],
    "contact_endpoint": "http://localhost:8001/mcp",
    "max_concurrent_matches": 2
  }
}
```

**Response: REFEREE_REGISTER_RESPONSE**

| Field | Type | Required |
|-------|------|----------|
| `message_type` | string | ✅ |
| `status` | string | ✅ |
| `referee_id` | string | ✅ |
| `reason` | string | ❌ |

**Example:**
```json
{
  "protocol": "league.v2",
  "message_type": "REFEREE_REGISTER_RESPONSE",
  "sender": "league_manager",
  "timestamp": "2025-01-15T09:00:01Z",
  "conversation_id": "conv-ref01-registration",
  "status": "ACCEPTED",
  "referee_id": "REF01",
  "reason": null
}
```

---

### 2.2 GAME_INVITATION
**Direction**: Referee → Player  
**When**: Match about to start  
**Timeout**: Player must respond within 5 seconds

**Fields:**

| Field | Type | Required |
|-------|------|----------|
| `message_type` | string | ✅ |
| `league_id` | string | ✅ |
| `round_id` | integer | ✅ |
| `match_id` | string | ✅ |
| `game_type` | string | ✅ |
| `role_in_match` | string | ✅ |
| `opponent_id` | string | ✅ |
| `conversation_id` | string | ✅ |

**Example:**
```json
{
  "protocol": "league.v2",
  "message_type": "GAME_INVITATION",
  "sender": "referee:REF01",
  "timestamp": "2025-01-15T10:30:00Z",
  "conversation_id": "conv-r1m1-001",
  "league_id": "league_2025_even_odd",
  "round_id": 1,
  "match_id": "R1M1",
  "game_type": "even_odd",
  "role_in_match": "PLAYER_A",
  "opponent_id": "P02"
}
```

---

### 2.3 CHOOSE_PARITY_CALL
**Direction**: Referee → Player  
**When**: Both players joined, game starts  
**Timeout**: Player must respond within 30 seconds

**Fields:**

| Field | Type | Required |
|-------|------|----------|
| `message_type` | string | ✅ |
| `match_id` | string | ✅ |
| `player_id` | string | ✅ |
| `game_type` | string | ✅ |
| `context` | object | ✅ |
| `deadline` | ISO-8601 | ✅ |

**Example:**
```json
{
  "protocol": "league.v2",
  "message_type": "CHOOSE_PARITY_CALL",
  "sender": "referee:REF01",
  "timestamp": "2025-01-15T10:30:05Z",
  "conversation_id": "conv-r1m1-001",
  "match_id": "R1M1",
  "player_id": "P01",
  "game_type": "even_odd",
  "context": {
    "opponent_id": "P02",
    "round_id": 1,
    "your_standings": {
      "wins": 2,
      "losses": 1,
      "draws": 0
    }
  },
  "deadline": "2025-01-15T10:30:30Z"
}
```

---

### 2.4 GAME_OVER
**Direction**: Referee → Both Players  
**When**: Game finished, winner determined

**Fields:**

| Field | Type | Required |
|-------|------|----------|
| `message_type` | string | ✅ |
| `match_id` | string | ✅ |
| `game_type` | string | ✅ |
| `game_result` | object | ✅ |

**game_result.status values**: `"WIN"` | `"DRAW"` | `"TECHNICAL_LOSS"`

**Example:**
```json
{
  "protocol": "league.v2",
  "message_type": "GAME_OVER",
  "sender": "referee:REF01",
  "timestamp": "2025-01-15T10:30:35Z",
  "conversation_id": "conv-r1m1-001",
  "match_id": "R1M1",
  "game_type": "even_odd",
  "game_result": {
    "status": "WIN",
    "winner_player_id": "P01",
    "drawn_number": 8,
    "number_parity": "even",
    "choices": {
      "P01": "even",
      "P02": "odd"
    },
    "reason": "P01 chose even, number was 8 (even)"
  }
}
```

---

### 2.5 MATCH_RESULT_REPORT
**Direction**: Referee → League Manager  
**When**: Game completed, reporting result

**Fields:**

| Field | Type | Required |
|-------|------|----------|
| `message_type` | string | ✅ |
| `league_id` | string | ✅ |
| `round_id` | integer | ✅ |
| `match_id` | string | ✅ |
| `game_type` | string | ✅ |
| `result` | object | ✅ |

**Example:**
```json
{
  "protocol": "league.v2",
  "message_type": "MATCH_RESULT_REPORT",
  "sender": "referee:REF01",
  "timestamp": "2025-01-15T10:30:36Z",
  "conversation_id": "conv-r1m1-001",
  "league_id": "league_2025_even_odd",
  "round_id": 1,
  "match_id": "R1M1",
  "game_type": "even_odd",
  "result": {
    "winner": "P01",
    "score": {
      "P01": 3,
      "P02": 0
    },
    "details": {
      "drawn_number": 8,
      "choices": {
        "P01": "even",
        "P02": "odd"
      }
    }
  }
}
```

**Response: MATCH_RESULT_ACK**

| Field | Type | Required |
|-------|------|----------|
| `protocol` | string | ✅ |
| `message_type` | string | ✅ |
| `match_id` | string | ✅ |
| `status` | string | ✅ |

**Example:**
```json
{
  "protocol": "league.v2",
  "message_type": "MATCH_RESULT_ACK",
  "sender": "league_manager",
  "timestamp": "2025-01-15T10:30:37Z",
  "conversation_id": "conv-r1m1-001",
  "match_id": "R1M1",
  "status": "recorded"
}
```

---

### 2.6 GAME_ERROR
**Direction**: Referee → Player  
**When**: Game-level error occurs

**Fields:**

| Field | Type | Required |
|-------|------|----------|
| `protocol` | string | ✅ |
| `message_type` | string | ✅ |
| `sender` | string | ✅ |
| `timestamp` | ISO-8601 | ✅ |
| `conversation_id` | string | ✅ |
| `match_id` | string | ✅ |
| `error_code` | string | ✅ |
| `error_description` | string | ✅ |
| `affected_player` | string | ✅ |
| `action_required` | string | ✅ |
| `retry_info` | object | ❌ |
| `consequence` | string | ❌ |

**Example:**
```json
{
  "protocol": "league.v2",
  "message_type": "GAME_ERROR",
  "sender": "referee:REF01",
  "timestamp": "2025-01-15T10:16:00Z",
  "conversation_id": "conv-r1m1-001",
  "match_id": "R1M1",
  "error_code": "E001",
  "error_description": "TIMEOUT_ERROR",
  "affected_player": "P02",
  "action_required": "CHOOSE_PARITY_RESPONSE",
  "retry_info": {
    "retry_count": 1,
    "max_retries": 3,
    "next_retry_at": "2025-01-15T10:16:02Z"
  },
  "consequence": "Technical loss if max retries exceeded"
}
```

---

## 3. PLAYER SOURCE

### 3.1 LEAGUE_REGISTER_REQUEST
**Direction**: Player → League Manager  
**Endpoint**: `POST /mcp`  
**When**: Player starts up

**Fields:**

| Field | Type | Required |
|-------|------|----------|
| `message_type` | string | ✅ |
| `player_meta` | object | ✅ |

**Example:**
```json
{
  "protocol": "league.v2",
  "message_type": "LEAGUE_REGISTER_REQUEST",
  "sender": "player:P01",
  "timestamp": "2025-01-15T09:01:00Z",
  "conversation_id": "conv-p01-registration",
  "player_meta": {
    "display_name": "Agent Alpha",
    "version": "1.0.0",
    "game_types": ["even_odd"],
    "contact_endpoint": "http://localhost:8101/mcp"
  }
}
```

**Response: LEAGUE_REGISTER_RESPONSE**

| Field | Type | Required |
|-------|------|----------|
| `message_type` | string | ✅ |
| `status` | string | ✅ |
| `player_id` | string | ✅ |
| `reason` | string | ❌ |

**Example:**
```json
{
  "protocol": "league.v2",
  "message_type": "LEAGUE_REGISTER_RESPONSE",
  "sender": "league_manager",
  "timestamp": "2025-01-15T09:01:01Z",
  "conversation_id": "conv-p01-registration",
  "status": "ACCEPTED",
  "player_id": "P01",
  "reason": null
}
```

---

### 3.2 GAME_JOIN_ACK
**Direction**: Player → Referee  
**When**: Received GAME_INVITATION  
**Timeout**: Must respond within 5 seconds

**Fields:**

| Field | Type | Required |
|-------|------|----------|
| `message_type` | string | ✅ |
| `match_id` | string | ✅ |
| `player_id` | string | ✅ |
| `arrival_timestamp` | ISO-8601 | ✅ |
| `accept` | boolean | ✅ |

**Example:**
```json
{
  "protocol": "league.v2",
  "message_type": "GAME_JOIN_ACK",
  "sender": "player:P01",
  "timestamp": "2025-01-15T10:30:02Z",
  "conversation_id": "conv-r1m1-001",
  "match_id": "R1M1",
  "player_id": "P01",
  "arrival_timestamp": "2025-01-15T10:30:02Z",
  "accept": true
}
```

---

### 3.3 CHOOSE_PARITY_RESPONSE
**Direction**: Player → Referee  
**When**: Received CHOOSE_PARITY_CALL  
**Timeout**: Must respond within 30 seconds

**Fields:**

| Field | Type | Required |
|-------|------|----------|
| `message_type` | string | ✅ |
| `match_id` | string | ✅ |
| `player_id` | string | ✅ |
| `parity_choice` | string | ✅ |

**Example:**
```json
{
  "protocol": "league.v2",
  "message_type": "CHOOSE_PARITY_RESPONSE",
  "sender": "player:P01",
  "timestamp": "2025-01-15T10:30:10Z",
  "conversation_id": "conv-r1m1-001",
  "match_id": "R1M1",
  "player_id": "P01",
  "parity_choice": "even"
}
```

---

### 3.4 LEAGUE_QUERY
**Direction**: Player/Referee → League Manager  
**Endpoint**: `POST /mcp`  
**When**: Agent needs league information

**Fields:**

| Field | Type | Required |
|-------|------|----------|
| `protocol` | string | ✅ |
| `message_type` | string | ✅ |
| `sender` | string | ✅ |
| `timestamp` | ISO-8601 | ✅ |
| `conversation_id` | string | ✅ |
| `auth_token` | string | ✅ |
| `league_id` | string | ✅ |
| `query_type` | string | ✅ |
| `query_params` | object | ❌ |

**Example:**
```json
{
  "protocol": "league.v2",
  "message_type": "LEAGUE_QUERY",
  "sender": "player:P01",
  "timestamp": "2025-01-15T14:00:00Z",
  "conversation_id": "conv-query-001",
  "auth_token": "tok_p01_abc123",
  "league_id": "league_2025_even_odd",
  "query_type": "GET_NEXT_MATCH",
  "query_params": {
    "player_id": "P01"
  }
}
```

**Response: LEAGUE_QUERY_RESPONSE** (see section 1.5)

---

## 4. LAUNCHER SOURCE

### 4.1 START_LEAGUE
**Direction**: Launcher → League Manager  
**Endpoint**: `POST /mcp`  
**When**: Starting the league

**Fields:**

| Field | Type | Required |
|-------|------|----------|
| `protocol` | string | ✅ |
| `message_type` | string | ✅ |
| `league_id` | string | ✅ |
| `sender` | string | ✅ |

**Example:**
```json
{
  "protocol": "league.v2",
  "message_type": "START_LEAGUE",
  "sender": "launcher",
  "timestamp": "2025-01-15T09:30:00Z",
  "conversation_id": "conv-start-league",
  "league_id": "league_2025_even_odd"
}
```

**Response: LEAGUE_STATUS**

| Field | Type | Required |
|-------|------|----------|
| `protocol` | string | ✅ |
| `message_type` | string | ✅ |
| `league_id` | string | ✅ |
| `status` | string | ✅ |
| `current_round` | integer | ✅ |
| `total_rounds` | integer | ✅ |
| `matches_completed` | integer | ✅ |

**Example:**
```json
{
  "protocol": "league.v2",
  "message_type": "LEAGUE_STATUS",
  "sender": "league_manager",
  "timestamp": "2025-01-15T09:30:01Z",
  "conversation_id": "conv-start-league",
  "league_id": "league_2025_even_odd",
  "status": "running",
  "current_round": 1,
  "total_rounds": 3,
  "matches_completed": 0
}
```

---

## 5. ERRORS

### Error Codes

| Code | Name | Description |
|------|------|-------------|
| `E001` | `TIMEOUT_ERROR` | Response timeout |
| `E003` | `MISSING_REQUIRED_FIELD` | Required field missing |
| `E004` | `INVALID_PARITY_CHOICE` | Invalid choice (not "even"/"odd") |
| `E005` | `PLAYER_NOT_REGISTERED` | Player not registered |
| `E009` | `CONNECTION_ERROR` | Connection failed |
| `E011` | `AUTH_TOKEN_MISSING` | Missing auth token |
| `E012` | `AUTH_TOKEN_INVALID` | Invalid auth token |
| `E021` | `INVALID_TIMESTAMP` | Invalid timestamp format |

---

### Response Timeouts

| Message Type | Timeout | Consequence |
|--------------|---------|-------------|
| `GAME_JOIN_ACK` | 5s | Player forfeits |
| `CHOOSE_PARITY_RESPONSE` | 30s | Technical loss |
| Registration | 10s | Registration fails |
| Default | 10s | Retry or error |
