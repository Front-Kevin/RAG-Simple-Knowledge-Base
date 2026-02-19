import { create } from 'zustand'

interface AppState {
  modelProvider: string
  chunkMode: string
  chunkSize: number
  overlap: number
  topK: number
  useRerank: boolean
  setModelProvider: (v: string) => void
  setChunkMode: (v: string) => void
  setChunkSize: (v: number) => void
  setOverlap: (v: number) => void
  setTopK: (v: number) => void
  setUseRerank: (v: boolean) => void
}

export const useAppStore = create<AppState>((set) => ({
  modelProvider: 'openai',
  chunkMode: 'sliding',
  chunkSize: 500,
  overlap: 100,
  topK: 5,
  useRerank: true,
  setModelProvider: (v) => set({ modelProvider: v }),
  setChunkMode: (v) => set({ chunkMode: v }),
  setChunkSize: (v) => set({ chunkSize: v }),
  setOverlap: (v) => set({ overlap: v }),
  setTopK: (v) => set({ topK: v }),
  setUseRerank: (v) => set({ useRerank: v }),
}))
