import type {
  APIGatewayProxyEventV2,
  APIGatewayProxyStructuredResultV2,
} from "aws-lambda";
import { getDataset } from "../shared/ddb";

/**
 * GET /v1/customers/{cid}/datasets/{dsid}/events
 *
 * MVP: returns dataset metadata only. Actual range retrieval over
 * compressed blobs comes after Spike 1 validates ratios.
 *
 * Once implemented, this will:
 * 1. Parse start_ts/end_ts from query string
 * 2. List compressed blobs in that range (time-bucketed prefix scan)
 * 3. Stream + decompress each blob, filter events in range
 * 4. Return NDJSON
 */
export const handler = async (
  event: APIGatewayProxyEventV2,
): Promise<APIGatewayProxyStructuredResultV2> => {
  const cid = event.pathParameters?.cid;
  const dsid = event.pathParameters?.dsid;
  if (!cid || !dsid) {
    return json(400, { error: "missing path parameters" });
  }

  const ds = await getDataset({ cid, dsid });
  if (!ds) {
    return json(404, { error: "dataset not found" });
  }

  // Spike 1 stub: return metadata, not events.
  return json(202, {
    status: ds.status,
    current_model_version: ds.current_model_version,
    raw_bytes_held: ds.raw_bytes_held,
    compressed_bytes: ds.compressed_bytes,
    total_events_ingested: ds.total_events_ingested,
    note: "range retrieval not implemented in MVP; metadata only",
  });
};

function json(
  statusCode: number,
  body: unknown,
): APIGatewayProxyStructuredResultV2 {
  return {
    statusCode,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}
