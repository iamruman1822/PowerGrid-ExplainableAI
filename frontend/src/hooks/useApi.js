import { useState, useEffect, useCallback } from 'react';
import { API_BASE } from '../utils/constants';

/**
 * Custom hook for fetching data from the Flask backend.
 *
 * @param {string} endpoint - API endpoint path (e.g., '/dashboard')
 * @param {object} options  - { autoFetch, method, body }
 */
export function useApi(endpoint, options = {}) {
  const { autoFetch = true, method = 'GET', body = null } = options;
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async (overrideBody = null) => {
    setLoading(true);
    setError(null);
    try {
      const config = {
        method: overrideBody || body ? 'POST' : method,
        headers: { 'Content-Type': 'application/json' },
      };
      if (overrideBody || body) {
        config.body = JSON.stringify(overrideBody || body);
      }
      const res = await fetch(`${API_BASE}${endpoint}`, config);
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      const json = await res.json();
      setData(json);
      return json;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  }, [endpoint, method, body]);

  useEffect(() => {
    if (autoFetch && method === 'GET') {
      fetchData();
    }
  }, [autoFetch, method, fetchData]);

  return { data, loading, error, refetch: fetchData };
}
