'use client';

import React, { useState, useEffect } from 'react';
import { useLeaderboardWebSocket } from '../utils/websocket';

// Types for leaderboard entries
export interface LeaderboardEntry {
  userId: string;
  name: string;
  rank: number;
  score: number;
  avatarUrl?: string;
  isCurrentUser?: boolean;
  change?: number; // Change in rank since last update
  badges?: string[]; // Achievement badges
  lastActive?: string; // ISO date string
}

interface LeaderboardProps {
  boardId: string;
  title?: string;
  authToken?: string;
  initialEntries?: LeaderboardEntry[];
  maxEntries?: number;
  showAvatar?: boolean;
  showRankChange?: boolean;
  showBadges?: boolean;
  showLastActive?: boolean;
  onEntryClick?: (entry: LeaderboardEntry) => void;
  currentUserId?: string;
  isLoading?: boolean;
  className?: string;
}

export default function Leaderboard({
  boardId,
  title = 'Leaderboard',
  authToken,
  initialEntries = [],
  maxEntries = 10,
  showAvatar = true,
  showRankChange = true,
  showBadges = false,
  showLastActive = false,
  onEntryClick,
  currentUserId,
  isLoading = false,
  className = '',
}: LeaderboardProps) {
  const [entries, setEntries] = useState<LeaderboardEntry[]>(initialEntries);
  const [highlightedEntryId, setHighlightedEntryId] = useState<string | null>(null);
  
  // Connect to WebSocket for real-time updates
  const { 
    status: wsStatus, 
    lastMessage 
  } = useLeaderboardWebSocket(boardId, authToken);
  
  // Handle initial entries
  useEffect(() => {
    if (initialEntries.length > 0) {
      // Mark current user
      const updatedEntries = initialEntries.map(entry => ({
        ...entry,
        isCurrentUser: entry.userId === currentUserId
      }));
      setEntries(updatedEntries);
    }
  }, [initialEntries, currentUserId]);
  
  // Handle WebSocket updates
  useEffect(() => {
    if (lastMessage && lastMessage.type === 'leaderboard_update') {
      const newEntries = lastMessage.data;
      
      if (Array.isArray(newEntries)) {
        // Calculate rank changes
        const updatedEntries = newEntries.map(newEntry => {
          const oldEntry = entries.find(e => e.userId === newEntry.userId);
          const rankChange = oldEntry ? oldEntry.rank - newEntry.rank : 0;
          
          // Highlight entries with rank changes
          if (rankChange !== 0) {
            setHighlightedEntryId(newEntry.userId);
            // Clear highlight after animation
            setTimeout(() => setHighlightedEntryId(null), 3000);
          }
          
          return {
            ...newEntry,
            change: rankChange,
            isCurrentUser: newEntry.userId === currentUserId
          };
        });
        
        setEntries(updatedEntries);
      }
    }
  }, [lastMessage, entries, currentUserId]);
  
  // Format time since last active
  const formatTimeSince = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);
    
    if (diffInSeconds < 60) {
      return 'just now';
    } else if (diffInSeconds < 3600) {
      const minutes = Math.floor(diffInSeconds / 60);
      return `${minutes}m ago`;
    } else if (diffInSeconds < 86400) {
      const hours = Math.floor(diffInSeconds / 3600);
      return `${hours}h ago`;
    } else {
      const days = Math.floor(diffInSeconds / 86400);
      return `${days}d ago`;
    }
  };

  // Get rank change indicator
  const getRankChangeIndicator = (change: number | undefined) => {
    if (!change || change === 0) return null;
    
    if (change > 0) {
      return (
        <span className="inline-flex items-center text-green-600 dark:text-green-400">
          <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
          </svg>
          {change}
        </span>
      );
    } else {
      return (
        <span className="inline-flex items-center text-red-600 dark:text-red-400">
          <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
          </svg>
          {Math.abs(change)}
        </span>
      );
    }
  };
  
  // Get medal for top rankings
  const getMedalForRank = (rank: number) => {
    if (rank === 1) {
      return (
        <span className="text-yellow-500 font-bold" title="1st Place">
          🥇
        </span>
      );
    } else if (rank === 2) {
      return (
        <span className="text-gray-400 font-bold" title="2nd Place">
          🥈
        </span>
      );
    } else if (rank === 3) {
      return (
        <span className="text-amber-600 font-bold" title="3rd Place">
          🥉
        </span>
      );
    }
    
    return <span className="text-gray-600 dark:text-gray-400">{rank}</span>;
  };

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden ${className}`}>
      {/* Header */}
      <div className="px-4 py-3 bg-gray-50 dark:bg-gray-700 border-b border-gray-200 dark:border-gray-600">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
            {title}
          </h3>
          
          <div className="flex items-center">
            <span className={`h-2 w-2 rounded-full mr-2 ${
              wsStatus === 'open' 
                ? 'bg-green-500' 
                : wsStatus === 'connecting' || wsStatus === 'reconnecting'
                ? 'bg-yellow-500'
                : 'bg-red-500'
            }`} />
            <span className="text-xs text-gray-500 dark:text-gray-400">
              {wsStatus === 'open' 
                ? 'Live' 
                : wsStatus === 'connecting' || wsStatus === 'reconnecting'
                ? 'Connecting...'
                : 'Offline'}
            </span>
          </div>
        </div>
      </div>
      
      {/* Leaderboard body */}
      <div>
        {isLoading ? (
          <div className="p-6 text-center">
            <div className="animate-spin inline-block w-8 h-8 border-4 border-gray-300 border-t-blue-600 rounded-full" />
            <p className="mt-2 text-gray-600 dark:text-gray-300">Loading leaderboard...</p>
          </div>
        ) : entries.length === 0 ? (
          <div className="p-6 text-center text-gray-500 dark:text-gray-400">
            <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <p className="mt-2">No entries yet</p>
          </div>
        ) : (
          <ul className="divide-y divide-gray-200 dark:divide-gray-700">
            {entries
              .slice(0, maxEntries)
              .map((entry) => (
                <li 
                  key={entry.userId}
                  onClick={() => onEntryClick && onEntryClick(entry)}
                  className={`
                    px-4 py-3 
                    ${onEntryClick ? 'cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700' : ''} 
                    ${entry.isCurrentUser ? 'bg-blue-50 dark:bg-blue-900/20' : ''}
                    ${highlightedEntryId === entry.userId ? 'animate-pulse bg-yellow-50 dark:bg-yellow-900/20' : ''}
                    transition-colors duration-300
                  `}
                >
                  <div className="flex items-center">
                    {/* Rank */}
                    <div className="flex-shrink-0 w-8 text-center">
                      {getMedalForRank(entry.rank)}
                    </div>
                    
                    {/* Avatar */}
                    {showAvatar && (
                      <div className="flex-shrink-0 h-10 w-10">
                        {entry.avatarUrl ? (
                          <img 
                            className="h-10 w-10 rounded-full object-cover border-2 border-gray-200 dark:border-gray-700" 
                            src={entry.avatarUrl} 
                            alt={entry.name}
                          />
                        ) : (
                          <div className="h-10 w-10 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center text-gray-500 dark:text-gray-300 font-medium">
                            {entry.name.charAt(0).toUpperCase()}
                          </div>
                        )}
                      </div>
                    )}
                    
                    {/* Name and score */}
                    <div className={`min-w-0 flex-1 ${showAvatar ? 'ml-4' : ''}`}>
                      <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                        {entry.name}
                        {entry.isCurrentUser && (
                          <span className="ml-2 text-xs bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 px-1.5 py-0.5 rounded">
                            You
                          </span>
                        )}
                      </p>
                      
                      <div className="flex items-center text-sm text-gray-500 dark:text-gray-400">
                        <span className="font-medium">{entry.score.toLocaleString()} pts</span>
                        
                        {showRankChange && entry.change !== undefined && (
                          <span className="ml-2 text-xs">
                            {getRankChangeIndicator(entry.change)}
                          </span>
                        )}
                      </div>
                    </div>
                    
                    {/* Badges */}
                    {showBadges && entry.badges && entry.badges.length > 0 && (
                      <div className="ml-2 flex-shrink-0 flex">
                        {entry.badges.slice(0, 3).map((badge, index) => (
                          <span 
                            key={index} 
                            className="inline-block ml-1" 
                            title={badge}
                          >
                            {badge}
                          </span>
                        ))}
                        {entry.badges.length > 3 && (
                          <span className="ml-1 text-xs text-gray-500 dark:text-gray-400">
                            +{entry.badges.length - 3}
                          </span>
                        )}
                      </div>
                    )}
                    
                    {/* Last active */}
                    {showLastActive && entry.lastActive && (
                      <div className="ml-2 flex-shrink-0 text-xs text-gray-500 dark:text-gray-400">
                        {formatTimeSince(entry.lastActive)}
                      </div>
                    )}
                  </div>
                </li>
              ))}
          </ul>
        )}
      </div>
    </div>
  );
} 