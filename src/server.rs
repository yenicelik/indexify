use crate::data_repository_manager::DataRepositoryManager;
use crate::index::IndexManager;
use crate::persistence::{
    ContentType, DataConnector, DataRepository, Extractor, ExtractorType, SourceType, Text,
};
use crate::text_splitters::TextSplitterKind;
use crate::{
    CreateIndexParams, EmbeddingRouter, IndexDistance, MemoryManager, Message, ServerConfig,
};
use strum_macros::Display;

use anyhow::Result;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::{extract::State, routing::get, routing::post, Json, Router};
use pyo3::Python;
use tokio::signal;
use tracing::info;

use serde::{Deserialize, Serialize};
use smart_default::SmartDefault;
use std::collections::HashMap;

use std::net::SocketAddr;
use std::str::FromStr;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "extractor_type")]
enum ApiExtractorType {
    #[serde(rename = "embedding")]
    Embedding {
        model: String,
        //TODO: Rename this to distance
        distance: ApiIndexDistance,
        text_splitter: ApiTextSplitterKind,
    },
}

impl From<ExtractorType> for ApiExtractorType {
    fn from(value: ExtractorType) -> Self {
        match value {
            ExtractorType::Embedding {
                model,
                text_splitter,
                distance,
            } => ApiExtractorType::Embedding {
                model,
                distance: distance.into(),
                text_splitter: text_splitter.into(),
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "extractor")]
struct ApiExtractor {
    pub name: String,
    pub extractor_type: ApiExtractorType,
}

impl From<Extractor> for ApiExtractor {
    fn from(value: Extractor) -> Self {
        Self {
            name: value.name,
            extractor_type: value.extractor_type.into(),
        }
    }
}

impl From<ApiExtractor> for Extractor {
    fn from(val: ApiExtractor) -> Self {
        Extractor {
            name: val.name,
            extractor_type: match val.extractor_type {
                ApiExtractorType::Embedding {
                    model,
                    distance,
                    text_splitter,
                } => ExtractorType::Embedding {
                    model,
                    distance: distance.into(),
                    text_splitter: text_splitter.into(),
                },
            },
        }
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
struct ApiDataRepository {
    pub name: String,
    pub extractors: Vec<ApiExtractor>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl From<DataRepository> for ApiDataRepository {
    fn from(value: DataRepository) -> Self {
        let ap_extractors = value.extractors.into_iter().map(|e| e.into()).collect();
        ApiDataRepository {
            name: value.name,
            extractors: ap_extractors,
            metadata: value.metadata,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "source_type")]
pub enum ApiSourceType {
    #[serde(rename = "google_contact")]
    GoogleContact { metadata: Option<String> },
    #[serde(rename = "gmail")]
    Gmail { metadata: Option<String> },
}

impl From<ApiSourceType> for SourceType {
    fn from(value: ApiSourceType) -> Self {
        match value {
            ApiSourceType::GoogleContact { metadata } => SourceType::GoogleContact { metadata },
            ApiSourceType::Gmail { metadata } => SourceType::Gmail { metadata },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "content_type")]
pub enum ApiContentType {
    #[serde(rename = "document")]
    Document,
}

impl From<ApiContentType> for ContentType {
    fn from(value: ApiContentType) -> Self {
        match value {
            ApiContentType::Document => ContentType::Document,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ApiDataConnector {
    pub source: ApiSourceType,
    pub content_type: ApiContentType,
}

impl From<ApiDataConnector> for DataConnector {
    fn from(value: ApiDataConnector) -> Self {
        Self {
            source: value.source.into(),
            content_type: value.content_type.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, SmartDefault)]
struct SyncRepository {
    pub name: String,
    pub extractors: Vec<ApiExtractor>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub data_connectors: Vec<ApiDataConnector>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SyncRepositoryResponse {}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GetRepository {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GetRepositoryResponse {
    pub repository: ApiDataRepository,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ListRepositories {}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ListRepositoriesResponse {
    pub repositories: Vec<ApiDataRepository>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GenerateEmbeddingRequest {
    /// Input texts for which embeddings will be generated.
    inputs: Vec<String>,
    /// Name of the model to use for generating embeddings.
    model: String,
}

/// Response payload for generating text embeddings.
#[derive(Debug, Serialize, Deserialize)]
struct GenerateEmbeddingResponse {
    embeddings: Option<Vec<Vec<f32>>>,
}

/// An embedding model and its properties.
#[derive(Debug, Serialize, Deserialize)]
struct EmbeddingModel {
    /// Name of the embedding model.
    name: String,
    /// Number of dimensions in the embeddings generated by this model.
    dimensions: u64,
}

/// Response payload for listing available embedding models.
#[derive(Debug, Serialize, Deserialize)]
struct ListEmbeddingModelsResponse {
    /// List of available embedding models.
    models: Vec<EmbeddingModel>,
}

#[derive(SmartDefault, Debug, Serialize, Deserialize, strum::Display, Clone)]
#[strum(serialize_all = "snake_case")]
enum ApiTextSplitterKind {
    // Do not split text.
    #[serde(rename = "none")]
    None,

    /// Split text by new lines.
    #[default]
    #[serde(rename = "new_line")]
    NewLine,

    /// Split a document across the regex boundary
    #[serde(rename = "regex")]
    Regex { pattern: String },
}

impl From<TextSplitterKind> for ApiTextSplitterKind {
    fn from(value: TextSplitterKind) -> Self {
        match value {
            TextSplitterKind::Noop => ApiTextSplitterKind::None,
            TextSplitterKind::NewLine => ApiTextSplitterKind::NewLine,
            TextSplitterKind::Regex { pattern } => ApiTextSplitterKind::Regex { pattern },
        }
    }
}

impl From<ApiTextSplitterKind> for TextSplitterKind {
    fn from(val: ApiTextSplitterKind) -> Self {
        match val {
            ApiTextSplitterKind::None => TextSplitterKind::Noop,
            ApiTextSplitterKind::NewLine => TextSplitterKind::NewLine,
            ApiTextSplitterKind::Regex { pattern } => TextSplitterKind::Regex { pattern },
        }
    }
}

#[derive(Display, Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename = "distance")]
enum ApiIndexDistance {
    #[serde(rename = "dot")]
    #[strum(serialize = "dot")]
    #[default]
    Dot,

    #[serde(rename = "cosine")]
    #[strum(serialize = "cosine")]
    Cosine,

    #[serde(rename = "euclidean")]
    #[strum(serialize = "euclidean")]
    Euclidean,
}

impl From<ApiIndexDistance> for IndexDistance {
    fn from(value: ApiIndexDistance) -> Self {
        match value {
            ApiIndexDistance::Dot => IndexDistance::Dot,
            ApiIndexDistance::Cosine => IndexDistance::Cosine,
            ApiIndexDistance::Euclidean => IndexDistance::Euclidean,
        }
    }
}

impl From<IndexDistance> for ApiIndexDistance {
    fn from(val: IndexDistance) -> Self {
        match val {
            IndexDistance::Dot => ApiIndexDistance::Dot,
            IndexDistance::Cosine => ApiIndexDistance::Cosine,
            IndexDistance::Euclidean => ApiIndexDistance::Euclidean,
        }
    }
}

/// Request payload for creating a new vector index.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
struct IndexCreateRequest {
    name: String,
    embedding_model: String,
    distance: ApiIndexDistance,
    text_splitter: ApiTextSplitterKind,

    hash_on: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct IndexCreateResponse {}

struct IndexCreationArgs {
    index_params: CreateIndexParams,
    text_splitter: TextSplitterKind,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Document {
    pub text: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AddTextsRequest {
    index: String,
    documents: Vec<Document>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct IndexAdditionResponse {
    sequence: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct SearchRequest {
    index: String,
    query: String,
    k: u64,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct CreateMemorySessionRequest {
    session_id: Option<String>,
    index_args: IndexCreateRequest,
    metadata: Option<HashMap<String, String>>,
}

#[derive(Serialize, Deserialize)]
struct CreateMemorySessionResponse {
    session_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct MemorySessionAddRequest {
    session_id: String,
    messages: Vec<Message>,
}

#[derive(Serialize, Deserialize)]
struct MemorySessionAddResponse {}

#[derive(Debug, Serialize, Deserialize)]
struct MemorySessionRetrieveRequest {
    session_id: String,
}

#[derive(Serialize, Deserialize)]
struct MemorySessionRetrieveResponse {
    messages: Vec<Message>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MemorySessionSearchRequest {
    session_id: String,
    query: String,
    k: u64,
}

#[derive(Serialize, Deserialize)]
struct MemorySessionSearchResponse {
    messages: Vec<Message>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct DocumentFragment {
    text: String,
    metadata: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct IndexSearchResponse {
    results: Vec<DocumentFragment>,
}

pub struct IndexifyAPIError {
    status_code: StatusCode,
    message: String,
}

impl IndexifyAPIError {
    fn new(status_code: StatusCode, message: String) -> Self {
        Self {
            status_code,
            message,
        }
    }
}

impl IntoResponse for IndexifyAPIError {
    fn into_response(self) -> Response {
        (self.status_code, self.message).into_response()
    }
}

#[derive(Clone)]
pub struct IndexEndpointState {
    index_manager: Arc<IndexManager>,
    embedding_router: Arc<EmbeddingRouter>,
}

#[derive(Clone)]
pub struct MemoryEndpointState {
    memory_manager: Arc<MemoryManager>,
    embedding_router: Arc<EmbeddingRouter>,
}

#[derive(Clone)]
pub struct RepositoryEndpointState {
    repository_manager: Arc<DataRepositoryManager>,
}

pub struct Server {
    addr: SocketAddr,
    config: Arc<ServerConfig>,
}
impl Server {
    pub fn new(config: Arc<super::server_config::ServerConfig>) -> Result<Self> {
        let addr: SocketAddr = config.listen_addr.parse()?;
        Ok(Self { addr, config })
    }

    pub async fn run(&self) -> Result<()> {
        let embedding_router = Arc::new(EmbeddingRouter::new(self.config.clone())?);
        let repository_manager = DataRepositoryManager::new(&self.config.db_url).await?;
        let repository_endpoint_state = RepositoryEndpointState {
            repository_manager: Arc::new(repository_manager),
        };
        let index_manager = Arc::new(
            IndexManager::new(
                self.config.index_config.clone(),
                embedding_router.clone(),
                self.config.db_url.clone(),
            )
            .await?,
        );
        let memory_manager = Arc::new(MemoryManager::new(index_manager.clone()).await?);
        let index_state = IndexEndpointState {
            index_manager,
            embedding_router: embedding_router.clone(),
        };
        let memory_state = MemoryEndpointState {
            memory_manager: memory_manager.clone(),
            embedding_router: embedding_router.clone(),
        };
        let app = Router::new()
            .route("/", get(root))
            .route(
                "/embeddings/models",
                get(list_embedding_models).with_state(embedding_router.clone()),
            )
            .route(
                "/embeddings/generate",
                get(generate_embedding).with_state(embedding_router.clone()),
            )
            .route(
                "/index/create",
                post(index_create).with_state(index_state.clone()),
            )
            .route(
                "/index/add",
                post(add_texts).with_state(index_state.clone()),
            )
            .route(
                "/index/search",
                get(index_search).with_state(index_state.clone()),
            )
            .route(
                "/memory/create",
                post(create_memory_session).with_state(memory_state.clone()),
            )
            .route(
                "/memory/add",
                post(add_to_memory_session).with_state(memory_manager.clone()),
            )
            .route(
                "/memory/get",
                get(get_from_memory_session).with_state(memory_manager.clone()),
            )
            .route(
                "/memory/search",
                get(search_memory_session).with_state(memory_manager.clone()),
            )
            .route(
                "/repository/sync",
                post(sync_repository).with_state(repository_endpoint_state.clone()),
            )
            .route(
                "/repository/list",
                get(list_repositories).with_state(repository_endpoint_state.clone()),
            )
            .route(
                "/repository/get",
                get(get_repository).with_state(repository_endpoint_state.clone()),
            );
        info!("server is listening at addr {:?}", &self.addr.to_string());
        axum::Server::bind(&self.addr)
            .serve(app.into_make_service())
            .with_graceful_shutdown(shutdown_signal())
            .await?;
        Ok(())
    }
}

async fn root() -> &'static str {
    "Indexify Server"
}

#[axum_macros::debug_handler]
async fn sync_repository(
    State(state): State<RepositoryEndpointState>,
    Json(payload): Json<SyncRepository>,
) -> Result<Json<SyncRepositoryResponse>, IndexifyAPIError> {
    let extractors = payload
        .extractors
        .clone()
        .into_iter()
        .map(|e| e.into())
        .collect();
    let data_connectors = payload
        .data_connectors
        .clone()
        .into_iter()
        .map(|dc| dc.into())
        .collect();
    let data_repository = &DataRepository {
        name: payload.name.clone(),
        extractors,
        data_connectors,
        metadata: payload.metadata.clone(),
    };
    state
        .repository_manager
        .sync(data_repository)
        .await
        .map_err(|e| {
            IndexifyAPIError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to sync repository: {}", e),
            )
        })?;
    Ok(Json(SyncRepositoryResponse {}))
}

async fn list_repositories(
    State(state): State<RepositoryEndpointState>,
    _payload: Option<Json<ListRepositories>>,
) -> Result<Json<ListRepositoriesResponse>, IndexifyAPIError> {
    let repositories = state
        .repository_manager
        .list_repositories()
        .await
        .map_err(|e| {
            IndexifyAPIError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to list repositories: {}", e),
            )
        })?;
    let data_repos = repositories.into_iter().map(|r| r.into()).collect();
    Ok(Json(ListRepositoriesResponse {
        repositories: data_repos,
    }))
}

async fn get_repository(
    State(state): State<RepositoryEndpointState>,
    Json(payload): Json<GetRepository>,
) -> Result<Json<GetRepositoryResponse>, IndexifyAPIError> {
    let data_repo = state
        .repository_manager
        .get(payload.name)
        .await
        .map_err(|e| {
            IndexifyAPIError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to get repository: {}", e),
            )
        })?;
    Ok(Json(GetRepositoryResponse {
        repository: data_repo.into(),
    }))
}

async fn get_index_creation_args(
    embedding_router: Arc<EmbeddingRouter>,
    payload: IndexCreateRequest,
) -> Result<IndexCreationArgs, IndexifyAPIError> {
    let model = embedding_router
        .get_model(payload.embedding_model)
        .map_err(|e| IndexifyAPIError::new(StatusCode::BAD_REQUEST, e.to_string()))?;
    let index_params = CreateIndexParams {
        name: payload.name,
        vector_dim: model.dimensions(),
        distance: payload.distance.into(),
        unique_params: payload.hash_on,
    };
    let splitter_kind = TextSplitterKind::from_str(&payload.text_splitter.to_string()).unwrap();
    Ok(IndexCreationArgs {
        index_params,
        text_splitter: splitter_kind,
    })
}

#[axum_macros::debug_handler]
async fn index_create(
    State(state): State<IndexEndpointState>,
    Json(payload): Json<IndexCreateRequest>,
) -> Result<Json<IndexCreateResponse>, IndexifyAPIError> {
    let args = get_index_creation_args(state.embedding_router.clone(), payload.clone()).await?;

    let result = state
        .index_manager
        .create_index(
            args.index_params,
            payload.embedding_model,
            args.text_splitter,
        )
        .await;

    if let Err(err) = result {
        return Err(IndexifyAPIError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            err.to_string(),
        ));
    }
    Ok(Json(IndexCreateResponse {}))
}

#[axum_macros::debug_handler]
async fn add_texts(
    State(state): State<IndexEndpointState>,
    Json(payload): Json<AddTextsRequest>,
) -> Result<Json<IndexAdditionResponse>, IndexifyAPIError> {
    let may_be_index = state
        .index_manager
        .load(payload.index)
        .await
        .map_err(|e| IndexifyAPIError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let index = may_be_index.ok_or(IndexifyAPIError::new(
        StatusCode::BAD_REQUEST,
        "index doesn't exist".into(),
    ))?;
    let texts = payload
        .documents
        .iter()
        .map(|d| Text {
            text: d.text.to_owned(),
            metadata: d.metadata.to_owned(),
        })
        .collect();
    let result = index.add_texts(texts).await;
    if let Err(err) = result {
        return Err(IndexifyAPIError::new(
            StatusCode::BAD_REQUEST,
            err.to_string(),
        ));
    }

    Ok(Json(IndexAdditionResponse::default()))
}

#[axum_macros::debug_handler]
async fn create_memory_session(
    State(state): State<MemoryEndpointState>,
    Json(payload): Json<CreateMemorySessionRequest>,
) -> Result<Json<CreateMemorySessionResponse>, IndexifyAPIError> {
    let args =
        get_index_creation_args(state.embedding_router.clone(), payload.index_args.clone()).await?;

    let session_id = state
        .memory_manager
        .create_session_index(
            payload.session_id,
            args.index_params,
            payload.index_args.embedding_model,
            args.text_splitter,
            payload.metadata.unwrap(),
        )
        .await
        .map_err(|e| IndexifyAPIError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(CreateMemorySessionResponse { session_id }))
}

#[axum_macros::debug_handler]
async fn add_to_memory_session(
    State(memory_manager): State<Arc<MemoryManager>>,
    Json(payload): Json<MemorySessionAddRequest>,
) -> Result<Json<MemorySessionAddResponse>, IndexifyAPIError> {
    memory_manager
        .add_messages(&payload.session_id, payload.messages)
        .await
        .map_err(|e| IndexifyAPIError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(MemorySessionAddResponse {}))
}

#[axum_macros::debug_handler]
async fn get_from_memory_session(
    State(memory_manager): State<Arc<MemoryManager>>,
    Json(payload): Json<MemorySessionRetrieveRequest>,
) -> Result<Json<MemorySessionRetrieveResponse>, IndexifyAPIError> {
    let messages = memory_manager
        .retrieve_messages(payload.session_id)
        .await
        .map_err(|e| IndexifyAPIError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(MemorySessionRetrieveResponse { messages }))
}

#[axum_macros::debug_handler]
async fn search_memory_session(
    State(memory_manager): State<Arc<MemoryManager>>,
    Json(payload): Json<MemorySessionSearchRequest>,
) -> Result<Json<MemorySessionSearchResponse>, IndexifyAPIError> {
    let messages = memory_manager
        .search(&payload.session_id, payload.query, payload.k)
        .await
        .map_err(|e| IndexifyAPIError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(MemorySessionSearchResponse { messages }))
}

#[axum_macros::debug_handler]
async fn index_search(
    State(state): State<IndexEndpointState>,
    Json(query): Json<SearchRequest>,
) -> Result<Json<IndexSearchResponse>, IndexifyAPIError> {
    let try_index = state.index_manager.load(query.index.clone()).await;
    if let Err(err) = try_index {
        return Err(IndexifyAPIError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            err.to_string(),
        ));
    }
    if try_index.as_ref().unwrap().is_none() {
        return Err(IndexifyAPIError::new(
            StatusCode::BAD_REQUEST,
            "index does not exist".into(),
        ));
    }
    let index = try_index.unwrap().unwrap();
    let results = index.search(query.query, query.k).await;
    if let Err(err) = results {
        return Err(IndexifyAPIError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            err.to_string(),
        ));
    }
    let document_fragments: Vec<DocumentFragment> = results
        .unwrap()
        .iter()
        .map(|text| DocumentFragment {
            text: text.texts.to_owned(),
            metadata: text.metadata.to_owned(),
        })
        .collect();
    Ok(Json(IndexSearchResponse {
        results: document_fragments,
    }))
}

#[axum_macros::debug_handler]
async fn list_embedding_models(
    State(embedding_router): State<Arc<EmbeddingRouter>>,
) -> Json<ListEmbeddingModelsResponse> {
    let model_names = embedding_router.list_models();
    let mut models: Vec<EmbeddingModel> = Vec::new();
    for model_name in model_names {
        let model = embedding_router.get_model(model_name.clone()).unwrap();
        models.push(EmbeddingModel {
            name: model_name.clone(),
            dimensions: model.dimensions(),
        })
    }
    Json(ListEmbeddingModelsResponse { models })
}

#[axum_macros::debug_handler]
async fn generate_embedding(
    State(embedding_router): State<Arc<EmbeddingRouter>>,
    Json(payload): Json<GenerateEmbeddingRequest>,
) -> Result<Json<GenerateEmbeddingResponse>, IndexifyAPIError> {
    let try_embedding_generator = embedding_router.get_model(payload.model);
    if let Err(err) = &try_embedding_generator {
        return Err(IndexifyAPIError::new(
            StatusCode::NOT_ACCEPTABLE,
            err.to_string(),
        ));
    }
    let embeddings = try_embedding_generator
        .unwrap()
        .generate_embeddings(payload.inputs)
        .await;

    if let Err(err) = embeddings {
        return Err(IndexifyAPIError::new(
            StatusCode::EXPECTATION_FAILED,
            err.to_string(),
        ));
    }

    Ok(Json(GenerateEmbeddingResponse {
        embeddings: Some(embeddings.unwrap()),
    }))
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            let _ = Python::with_gil(|py| py.check_signals());
        },
        _ = terminate => {
            let _ = Python::with_gil(|py| py.check_signals());
        },
    }
    info!("signal received, shutting down server gracefully");
}
