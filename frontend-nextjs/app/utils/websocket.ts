/**
 * WebSocket Client Module
 * 
 * This module provides a WebSocket client for real-time communication with the TradeIQ backend.
 * It handles connection lifecycle, authentication, reconnection, and message routing.
 */

import { useState, useEffect, useCallback, useRef } from 'react';

// WebSocket message types
export interface WebSocketMessage {
  type: string;
  [key: string]: any;
}

// WebSocket connection options
export interface WebSocketOptions {
  /** Authorization token for authenticated connections */
  token?: string;
  
  /** Automatically reconnect on disconnection */
  autoReconnect?: boolean;
  
  /** Maximum reconnection attempts (0 for unlimited) */
  maxReconnectAttempts?: number;
  
  /** Initial reconnect delay in milliseconds */
  reconnectDelay?: number;
  
  /** Maximum reconnect delay in milliseconds */
  maxReconnectDelay?: number;
  
  /** Whether to connect immediately on hook initialization */
  connectImmediately?: boolean;
  
  /** Callback for connection open */
  onOpen?: (event: Event) => void;
  
  /** Callback for connection close */
  onClose?: (event: CloseEvent) => void;
  
  /** Callback for connection error */
  onError?: (event: Event) => void;
  
  /** Callback for incoming messages */
  onMessage?: (data: any) => void;
}

// Connection status enum
export enum ConnectionStatus {
  CONNECTING = 'connecting',
  OPEN = 'open',
  CLOSING = 'closing',
  CLOSED = 'closed',
  RECONNECTING = 'reconnecting',
  FAILED = 'failed',
}

// WebSocket hook return type
export interface UseWebSocketReturn {
  /** Current WebSocket connection status */
  status: ConnectionStatus;
  
  /** Send a message to the WebSocket server */
  sendMessage: (message: WebSocketMessage | string) => boolean;
  
  /** Join a specific group for group-based messaging */
  joinGroup: (group: string) => void;
  
  /** Leave a specific group */
  leaveGroup: (group: string) => void;
  
  /** Manually connect to the WebSocket server */
  connect: () => void;
  
  /** Manually disconnect from the WebSocket server */
  disconnect: () => void;
  
  /** Last received message */
  lastMessage: any;
  
  /** Last error that occurred */
  lastError: Error | null;
  
  /** Number of reconnection attempts made */
  reconnectAttempts: number;
}

/**
 * React hook for WebSocket communication with the TradeIQ backend.
 * 
 * @param url WebSocket endpoint URL
 * @param options Connection and behavior options
 * @returns WebSocket utilities and state
 */
export const useWebSocket = (
  url: string,
  options: WebSocketOptions = {}
): UseWebSocketReturn => {
  const {
    token,
    autoReconnect = true,
    maxReconnectAttempts = 5,
    reconnectDelay = 1000,
    maxReconnectDelay = 30000,
    connectImmediately = true,
    onOpen,
    onClose,
    onError,
    onMessage,
  } = options;

  // State and refs
  const [status, setStatus] = useState<ConnectionStatus>(ConnectionStatus.CLOSED);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const [lastError, setLastError] = useState<Error | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState<number>(0);
  
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  // Compute the full URL with token if provided
  const getFullUrl = useCallback(() => {
    const fullUrl = new URL(url, window.location.origin);
    if (token) {
      fullUrl.searchParams.append('token', token);
    }
    return fullUrl.toString();
  }, [url, token]);

  // Connect to WebSocket server
  const connect = useCallback(() => {
    // Clean up any existing connection
    if (socketRef.current) {
      // Fix the readyState type issue with type assertion
      const readyState = socketRef.current.readyState;
      if (readyState === WebSocket.OPEN || readyState === WebSocket.CONNECTING) {
        socketRef.current.close();
      }
    }
    
    try {
      setStatus(ConnectionStatus.CONNECTING);
      
      // Create new WebSocket connection
      const fullUrl = getFullUrl();
      const socket = new WebSocket(fullUrl);
      socketRef.current = socket;
      
      // Setup event handlers
      socket.onopen = (event) => {
        setStatus(ConnectionStatus.OPEN);
        setReconnectAttempts(0);
        if (onOpen) onOpen(event);
      };
      
      socket.onclose = (event) => {
        setStatus(ConnectionStatus.CLOSED);
        if (onClose) onClose(event);
        
        // Handle reconnection logic
        if (autoReconnect && (!maxReconnectAttempts || reconnectAttempts < maxReconnectAttempts)) {
          setStatus(ConnectionStatus.RECONNECTING);
          
          // Calculate exponential backoff for reconnect
          const delay = Math.min(
            reconnectDelay * Math.pow(1.5, reconnectAttempts),
            maxReconnectDelay
          );
          
          reconnectTimeoutRef.current = setTimeout(() => {
            setReconnectAttempts((prev) => prev + 1);
            connect();
          }, delay);
        } else if (reconnectAttempts >= maxReconnectAttempts) {
          setStatus(ConnectionStatus.FAILED);
        }
      };
      
      socket.onerror = (event) => {
        const error = new Error('WebSocket connection error');
        setLastError(error);
        if (onError) onError(event);
      };
      
      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);
          if (onMessage) onMessage(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
          setLastError(error instanceof Error ? error : new Error(String(error)));
        }
      };
    } catch (error) {
      setStatus(ConnectionStatus.FAILED);
      setLastError(error instanceof Error ? error : new Error(String(error)));
      
      // Attempt reconnection
      if (autoReconnect && (!maxReconnectAttempts || reconnectAttempts < maxReconnectAttempts)) {
        setStatus(ConnectionStatus.RECONNECTING);
        reconnectTimeoutRef.current = setTimeout(() => {
          setReconnectAttempts((prev) => prev + 1);
          connect();
        }, reconnectDelay);
      }
    }
  }, [
    getFullUrl,
    autoReconnect,
    maxReconnectAttempts,
    reconnectAttempts,
    reconnectDelay,
    maxReconnectDelay,
    onOpen,
    onClose,
    onError,
    onMessage,
  ]);

  // Disconnect from WebSocket server
  const disconnect = useCallback(() => {
    // Clear any reconnection timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    // Close the connection if it exists
    if (socketRef.current) {
      setStatus(ConnectionStatus.CLOSING);
      socketRef.current.close();
      setReconnectAttempts(0);
    }
  }, []);

  // Send a message to the server
  const sendMessage = useCallback(
    (message: WebSocketMessage | string): boolean => {
      if (
        !socketRef.current ||
        socketRef.current.readyState !== WebSocket.OPEN
      ) {
        return false;
      }
      
      try {
        const messageString = typeof message === 'string' ? message : JSON.stringify(message);
        socketRef.current.send(messageString);
        return true;
      } catch (error) {
        console.error('Error sending WebSocket message:', error);
        setLastError(error instanceof Error ? error : new Error(String(error)));
        return false;
      }
    },
    []
  );

  // Join a WebSocket group
  const joinGroup = useCallback(
    (group: string) => {
      sendMessage({
        type: 'join_group',
        group,
      });
    },
    [sendMessage]
  );

  // Leave a WebSocket group
  const leaveGroup = useCallback(
    (group: string) => {
      sendMessage({
        type: 'leave_group',
        group,
      });
    },
    [sendMessage]
  );

  // Connect on mount if configured to do so
  useEffect(() => {
    if (connectImmediately) {
      connect();
    }
    
    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, [connect, disconnect, connectImmediately]);

  // Return the WebSocket API
  return {
    status,
    sendMessage,
    joinGroup,
    leaveGroup,
    connect,
    disconnect,
    lastMessage,
    lastError,
    reconnectAttempts,
  };
};

/**
 * Hook to subscribe to a specific assessment's WebSocket events
 * 
 * @param assessmentId ID of the assessment
 * @param token User authentication token
 * @returns WebSocket connection and utilities
 */
export const useAssessmentWebSocket = (
  assessmentId: string,
  token: string,
  options: Partial<WebSocketOptions> = {}
): UseWebSocketReturn => {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || '';
  const wsUrl = `${baseUrl.replace(/^http/, 'ws')}/ws/assessment/${assessmentId}`;
  
  return useWebSocket(wsUrl, {
    token,
    autoReconnect: true,
    connectImmediately: true,
    ...options,
  });
};

/**
 * Hook to subscribe to a leaderboard's WebSocket events
 * 
 * @param boardId ID of the leaderboard
 * @param token Optional user authentication token
 * @returns WebSocket connection and utilities
 */
export const useLeaderboardWebSocket = (
  boardId: string,
  token?: string,
  options: Partial<WebSocketOptions> = {}
): UseWebSocketReturn => {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || '';
  const wsUrl = `${baseUrl.replace(/^http/, 'ws')}/ws/leaderboard/${boardId}`;
  
  return useWebSocket(wsUrl, {
    token,
    autoReconnect: true,
    connectImmediately: true,
    ...options,
  });
}; 