// src/features/video/components/trajectory-grid.tsx

"use client";

import { useEffect, useState } from "react";
import { fetchVideoTrajectory } from "../api/fetch-trajectory";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface TrajectoryGridProps {
  videoId: string;
}

export function TrajectoryGrid({ videoId }: TrajectoryGridProps) {
  const [data, setData] = useState<any[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;

    async function loadData() {
      try {
        setLoading(true);
        const rawTrajectory = await fetchVideoTrajectory(videoId);
        
        if (isMounted) {
          const trajectoryArray = Array.isArray(rawTrajectory) ? rawTrajectory : [];
          
          // Flatten the [mood, energy] coordinate array into individual keys for Recharts
          const formattedData = trajectoryArray.map((point: any) => ({
            timestamp: point.timestamp,
            chunk_id: point.chunk_id,
            mood: point.coordinate[0],
            energy: point.coordinate[1]
          }));

          setData(formattedData);
          setLoading(false);
        }
      } catch (err: any) {
        if (isMounted) {
          setError(err.message);
          setLoading(false);
        }
      }
    }

    loadData();

    return () => {
      isMounted = false;
    };
  }, [videoId]);

  if (loading) {
    return (
      <div className="h-64 flex flex-col items-center justify-center text-gray-500 border border-dashed border-gray-200 rounded-lg">
        <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mb-3"></div>
        <p className="text-sm">Rendering trajectory chart...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm">
        Failed to load chart data: {error}
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg text-yellow-700 text-sm">
        No cinematic trajectory data found.
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm h-[400px] flex flex-col">
      <h3 className="text-sm font-semibold text-gray-800 mb-4">Cinematic Trajectory</h3>
      
      <div className="flex-grow w-full h-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 5, right: 20, left: 0, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
            
            <XAxis 
              dataKey="timestamp" 
              name="Time (s)"
              unit="s"
              tick={{ fontSize: 12, fill: '#6b7280' }}
              tickLine={false}
              axisLine={{ stroke: '#e5e7eb' }}
            />
            
            <YAxis 
              tick={{ fontSize: 12, fill: '#6b7280' }}
              tickLine={false}
              axisLine={false}
            />
            
            <Tooltip 
              contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
              labelFormatter={(value) => `Time: ${value}s`}
            />
            <Legend wrapperStyle={{ paddingTop: '10px', fontSize: '14px' }} />
            
            <Line 
              type="monotone" 
              dataKey="mood" 
              name="Mood"
              stroke="#3b82f6" 
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 6 }} 
            />
            
            <Line 
              type="monotone" 
              dataKey="energy" 
              name="Energy"
              stroke="#f59e0b" 
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 6 }} 
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}