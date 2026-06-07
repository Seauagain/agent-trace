/** Built-in builder that emits one trace per request/completion. */

import { buildTraceFromCompletion } from "../recordUtils.js";
import { type CompletionSession, createTrajectory, type Trajectory } from "../models.js";
import { type TrajectoryBuilder, topLevelSchedulerMetadata } from "./base.js";

export class PerRequestBuilder implements TrajectoryBuilder {
  async build(session: CompletionSession): Promise<Trajectory> {
    if (session.completions.length === 0) {
      return createTrajectory({
        status: "ERROR",
        metadata: {
          builder: "per_request",
          session_id: session.session_id,
          task_metadata: { ...session.metadata },
          record_count: 0,
          ...topLevelSchedulerMetadata(session.metadata),
        },
        traces: [],
        error: "no completions",
      });
    }

    return createTrajectory({
      status: "COMPLETED",
      metadata: {
        builder: "per_request",
        session_id: session.session_id,
        task_id: session.task_id ?? null,
        api_type: session.api_type ?? null,
        model_requested: session.model_requested ?? null,
        model_used: session.model_used ?? null,
        record_count: session.completions.length,
        task_metadata: { ...session.metadata },
        trace_count: session.completions.length,
        ...topLevelSchedulerMetadata(session.metadata),
      },
      traces: session.completions.map((completion) => buildTraceFromCompletion(completion)),
    });
  }
}
