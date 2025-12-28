export interface VidResult {
  task_id: string;
  filename: string;
  video_url?: string;
  tracked_vehicles?: number;
}

export enum DemoState {
  IDLE,
  ERROR,
  IS_UPLOADING,
  IS_PROCESSING,
  SUCCESS,
}
