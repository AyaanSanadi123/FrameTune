// lib/api-client.ts

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function apiClient<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  const response = await fetch(url, options);

  if (!response.ok) {
    let errorMessage = `HTTP Error ${response.status}`;
    try {
      // FastAPI specifically returns errors in a {"detail": "message"} format
      const errorData = await response.json();
      errorMessage = errorData.detail || errorMessage;
    } catch {
      // Fallback if the server crashes and returns HTML/Text
      errorMessage = response.statusText || errorMessage;
    }
    
    throw new Error(errorMessage);
  }

  return response.json();
}