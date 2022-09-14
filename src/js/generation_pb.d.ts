// package: gooseai
// file: generation.proto

import * as jspb from "google-protobuf";

export class Token extends jspb.Message {
  hasText(): boolean;
  clearText(): void;
  getText(): string;
  setText(value: string): void;

  getId(): number;
  setId(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Token.AsObject;
  static toObject(includeInstance: boolean, msg: Token): Token.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Token, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Token;
  static deserializeBinaryFromReader(message: Token, reader: jspb.BinaryReader): Token;
}

export namespace Token {
  export type AsObject = {
    text: string,
    id: number,
  }
}

export class Tokens extends jspb.Message {
  clearTokensList(): void;
  getTokensList(): Array<Token>;
  setTokensList(value: Array<Token>): void;
  addTokens(value?: Token, index?: number): Token;

  hasTokenizerId(): boolean;
  clearTokenizerId(): void;
  getTokenizerId(): string;
  setTokenizerId(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Tokens.AsObject;
  static toObject(includeInstance: boolean, msg: Tokens): Tokens.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Tokens, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Tokens;
  static deserializeBinaryFromReader(message: Tokens, reader: jspb.BinaryReader): Tokens;
}

export namespace Tokens {
  export type AsObject = {
    tokensList: Array<Token.AsObject>,
    tokenizerId: string,
  }
}

export class Artifact extends jspb.Message {
  getId(): number;
  setId(value: number): void;

  getType(): ArtifactTypeMap[keyof ArtifactTypeMap];
  setType(value: ArtifactTypeMap[keyof ArtifactTypeMap]): void;

  getMime(): string;
  setMime(value: string): void;

  hasMagic(): boolean;
  clearMagic(): void;
  getMagic(): string;
  setMagic(value: string): void;

  hasBinary(): boolean;
  clearBinary(): void;
  getBinary(): Uint8Array | string;
  getBinary_asU8(): Uint8Array;
  getBinary_asB64(): string;
  setBinary(value: Uint8Array | string): void;

  hasText(): boolean;
  clearText(): void;
  getText(): string;
  setText(value: string): void;

  hasTokens(): boolean;
  clearTokens(): void;
  getTokens(): Tokens | undefined;
  setTokens(value?: Tokens): void;

  hasClassifier(): boolean;
  clearClassifier(): void;
  getClassifier(): ClassifierParameters | undefined;
  setClassifier(value?: ClassifierParameters): void;

  getIndex(): number;
  setIndex(value: number): void;

  getFinishReason(): FinishReasonMap[keyof FinishReasonMap];
  setFinishReason(value: FinishReasonMap[keyof FinishReasonMap]): void;

  getSeed(): number;
  setSeed(value: number): void;

  getDataCase(): Artifact.DataCase;
  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Artifact.AsObject;
  static toObject(includeInstance: boolean, msg: Artifact): Artifact.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Artifact, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Artifact;
  static deserializeBinaryFromReader(message: Artifact, reader: jspb.BinaryReader): Artifact;
}

export namespace Artifact {
  export type AsObject = {
    id: number,
    type: ArtifactTypeMap[keyof ArtifactTypeMap],
    mime: string,
    magic: string,
    binary: Uint8Array | string,
    text: string,
    tokens?: Tokens.AsObject,
    classifier?: ClassifierParameters.AsObject,
    index: number,
    finishReason: FinishReasonMap[keyof FinishReasonMap],
    seed: number,
  }

  export enum DataCase {
    DATA_NOT_SET = 0,
    BINARY = 5,
    TEXT = 6,
    TOKENS = 7,
    CLASSIFIER = 11,
  }
}

export class PromptParameters extends jspb.Message {
  hasInit(): boolean;
  clearInit(): void;
  getInit(): boolean;
  setInit(value: boolean): void;

  hasWeight(): boolean;
  clearWeight(): void;
  getWeight(): number;
  setWeight(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): PromptParameters.AsObject;
  static toObject(includeInstance: boolean, msg: PromptParameters): PromptParameters.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: PromptParameters, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): PromptParameters;
  static deserializeBinaryFromReader(message: PromptParameters, reader: jspb.BinaryReader): PromptParameters;
}

export namespace PromptParameters {
  export type AsObject = {
    init: boolean,
    weight: number,
  }
}

export class Prompt extends jspb.Message {
  hasParameters(): boolean;
  clearParameters(): void;
  getParameters(): PromptParameters | undefined;
  setParameters(value?: PromptParameters): void;

  hasText(): boolean;
  clearText(): void;
  getText(): string;
  setText(value: string): void;

  hasTokens(): boolean;
  clearTokens(): void;
  getTokens(): Tokens | undefined;
  setTokens(value?: Tokens): void;

  hasArtifact(): boolean;
  clearArtifact(): void;
  getArtifact(): Artifact | undefined;
  setArtifact(value?: Artifact): void;

  getPromptCase(): Prompt.PromptCase;
  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Prompt.AsObject;
  static toObject(includeInstance: boolean, msg: Prompt): Prompt.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Prompt, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Prompt;
  static deserializeBinaryFromReader(message: Prompt, reader: jspb.BinaryReader): Prompt;
}

export namespace Prompt {
  export type AsObject = {
    parameters?: PromptParameters.AsObject,
    text: string,
    tokens?: Tokens.AsObject,
    artifact?: Artifact.AsObject,
  }

  export enum PromptCase {
    PROMPT_NOT_SET = 0,
    TEXT = 2,
    TOKENS = 3,
    ARTIFACT = 4,
  }
}

export class AnswerMeta extends jspb.Message {
  hasGpuId(): boolean;
  clearGpuId(): void;
  getGpuId(): string;
  setGpuId(value: string): void;

  hasCpuId(): boolean;
  clearCpuId(): void;
  getCpuId(): string;
  setCpuId(value: string): void;

  hasNodeId(): boolean;
  clearNodeId(): void;
  getNodeId(): string;
  setNodeId(value: string): void;

  hasEngineId(): boolean;
  clearEngineId(): void;
  getEngineId(): string;
  setEngineId(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): AnswerMeta.AsObject;
  static toObject(includeInstance: boolean, msg: AnswerMeta): AnswerMeta.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: AnswerMeta, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): AnswerMeta;
  static deserializeBinaryFromReader(message: AnswerMeta, reader: jspb.BinaryReader): AnswerMeta;
}

export namespace AnswerMeta {
  export type AsObject = {
    gpuId: string,
    cpuId: string,
    nodeId: string,
    engineId: string,
  }
}

export class Answer extends jspb.Message {
  getAnswerId(): string;
  setAnswerId(value: string): void;

  getRequestId(): string;
  setRequestId(value: string): void;

  getReceived(): number;
  setReceived(value: number): void;

  getCreated(): number;
  setCreated(value: number): void;

  hasMeta(): boolean;
  clearMeta(): void;
  getMeta(): AnswerMeta | undefined;
  setMeta(value?: AnswerMeta): void;

  clearArtifactsList(): void;
  getArtifactsList(): Array<Artifact>;
  setArtifactsList(value: Array<Artifact>): void;
  addArtifacts(value?: Artifact, index?: number): Artifact;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Answer.AsObject;
  static toObject(includeInstance: boolean, msg: Answer): Answer.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Answer, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Answer;
  static deserializeBinaryFromReader(message: Answer, reader: jspb.BinaryReader): Answer;
}

export namespace Answer {
  export type AsObject = {
    answerId: string,
    requestId: string,
    received: number,
    created: number,
    meta?: AnswerMeta.AsObject,
    artifactsList: Array<Artifact.AsObject>,
  }
}

export class SamplerParameters extends jspb.Message {
  hasEta(): boolean;
  clearEta(): void;
  getEta(): number;
  setEta(value: number): void;

  hasSamplingSteps(): boolean;
  clearSamplingSteps(): void;
  getSamplingSteps(): number;
  setSamplingSteps(value: number): void;

  hasLatentChannels(): boolean;
  clearLatentChannels(): void;
  getLatentChannels(): number;
  setLatentChannels(value: number): void;

  hasDownsamplingFactor(): boolean;
  clearDownsamplingFactor(): void;
  getDownsamplingFactor(): number;
  setDownsamplingFactor(value: number): void;

  hasCfgScale(): boolean;
  clearCfgScale(): void;
  getCfgScale(): number;
  setCfgScale(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): SamplerParameters.AsObject;
  static toObject(includeInstance: boolean, msg: SamplerParameters): SamplerParameters.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: SamplerParameters, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): SamplerParameters;
  static deserializeBinaryFromReader(message: SamplerParameters, reader: jspb.BinaryReader): SamplerParameters;
}

export namespace SamplerParameters {
  export type AsObject = {
    eta: number,
    samplingSteps: number,
    latentChannels: number,
    downsamplingFactor: number,
    cfgScale: number,
  }
}

export class ConditionerParameters extends jspb.Message {
  hasVectorAdjustPrior(): boolean;
  clearVectorAdjustPrior(): void;
  getVectorAdjustPrior(): string;
  setVectorAdjustPrior(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ConditionerParameters.AsObject;
  static toObject(includeInstance: boolean, msg: ConditionerParameters): ConditionerParameters.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: ConditionerParameters, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ConditionerParameters;
  static deserializeBinaryFromReader(message: ConditionerParameters, reader: jspb.BinaryReader): ConditionerParameters;
}

export namespace ConditionerParameters {
  export type AsObject = {
    vectorAdjustPrior: string,
  }
}

export class ScheduleParameters extends jspb.Message {
  hasStart(): boolean;
  clearStart(): void;
  getStart(): number;
  setStart(value: number): void;

  hasEnd(): boolean;
  clearEnd(): void;
  getEnd(): number;
  setEnd(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ScheduleParameters.AsObject;
  static toObject(includeInstance: boolean, msg: ScheduleParameters): ScheduleParameters.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: ScheduleParameters, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ScheduleParameters;
  static deserializeBinaryFromReader(message: ScheduleParameters, reader: jspb.BinaryReader): ScheduleParameters;
}

export namespace ScheduleParameters {
  export type AsObject = {
    start: number,
    end: number,
  }
}

export class StepParameter extends jspb.Message {
  getScaledStep(): number;
  setScaledStep(value: number): void;

  hasSampler(): boolean;
  clearSampler(): void;
  getSampler(): SamplerParameters | undefined;
  setSampler(value?: SamplerParameters): void;

  hasSchedule(): boolean;
  clearSchedule(): void;
  getSchedule(): ScheduleParameters | undefined;
  setSchedule(value?: ScheduleParameters): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): StepParameter.AsObject;
  static toObject(includeInstance: boolean, msg: StepParameter): StepParameter.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: StepParameter, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): StepParameter;
  static deserializeBinaryFromReader(message: StepParameter, reader: jspb.BinaryReader): StepParameter;
}

export namespace StepParameter {
  export type AsObject = {
    scaledStep: number,
    sampler?: SamplerParameters.AsObject,
    schedule?: ScheduleParameters.AsObject,
  }
}

export class TransformType extends jspb.Message {
  hasDiffusion(): boolean;
  clearDiffusion(): void;
  getDiffusion(): DiffusionSamplerMap[keyof DiffusionSamplerMap];
  setDiffusion(value: DiffusionSamplerMap[keyof DiffusionSamplerMap]): void;

  hasUpscaler(): boolean;
  clearUpscaler(): void;
  getUpscaler(): UpscalerMap[keyof UpscalerMap];
  setUpscaler(value: UpscalerMap[keyof UpscalerMap]): void;

  getTypeCase(): TransformType.TypeCase;
  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): TransformType.AsObject;
  static toObject(includeInstance: boolean, msg: TransformType): TransformType.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: TransformType, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): TransformType;
  static deserializeBinaryFromReader(message: TransformType, reader: jspb.BinaryReader): TransformType;
}

export namespace TransformType {
  export type AsObject = {
    diffusion: DiffusionSamplerMap[keyof DiffusionSamplerMap],
    upscaler: UpscalerMap[keyof UpscalerMap],
  }

  export enum TypeCase {
    TYPE_NOT_SET = 0,
    DIFFUSION = 1,
    UPSCALER = 2,
  }
}

export class ImageParameters extends jspb.Message {
  hasHeight(): boolean;
  clearHeight(): void;
  getHeight(): number;
  setHeight(value: number): void;

  hasWidth(): boolean;
  clearWidth(): void;
  getWidth(): number;
  setWidth(value: number): void;

  clearSeedList(): void;
  getSeedList(): Array<number>;
  setSeedList(value: Array<number>): void;
  addSeed(value: number, index?: number): number;

  hasSamples(): boolean;
  clearSamples(): void;
  getSamples(): number;
  setSamples(value: number): void;

  hasSteps(): boolean;
  clearSteps(): void;
  getSteps(): number;
  setSteps(value: number): void;

  hasTransform(): boolean;
  clearTransform(): void;
  getTransform(): TransformType | undefined;
  setTransform(value?: TransformType): void;

  clearParametersList(): void;
  getParametersList(): Array<StepParameter>;
  setParametersList(value: Array<StepParameter>): void;
  addParameters(value?: StepParameter, index?: number): StepParameter;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ImageParameters.AsObject;
  static toObject(includeInstance: boolean, msg: ImageParameters): ImageParameters.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: ImageParameters, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ImageParameters;
  static deserializeBinaryFromReader(message: ImageParameters, reader: jspb.BinaryReader): ImageParameters;
}

export namespace ImageParameters {
  export type AsObject = {
    height: number,
    width: number,
    seedList: Array<number>,
    samples: number,
    steps: number,
    transform?: TransformType.AsObject,
    parametersList: Array<StepParameter.AsObject>,
  }
}

export class ClassifierConcept extends jspb.Message {
  getConcept(): string;
  setConcept(value: string): void;

  hasThreshold(): boolean;
  clearThreshold(): void;
  getThreshold(): number;
  setThreshold(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ClassifierConcept.AsObject;
  static toObject(includeInstance: boolean, msg: ClassifierConcept): ClassifierConcept.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: ClassifierConcept, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ClassifierConcept;
  static deserializeBinaryFromReader(message: ClassifierConcept, reader: jspb.BinaryReader): ClassifierConcept;
}

export namespace ClassifierConcept {
  export type AsObject = {
    concept: string,
    threshold: number,
  }
}

export class ClassifierCategory extends jspb.Message {
  getName(): string;
  setName(value: string): void;

  clearConceptsList(): void;
  getConceptsList(): Array<ClassifierConcept>;
  setConceptsList(value: Array<ClassifierConcept>): void;
  addConcepts(value?: ClassifierConcept, index?: number): ClassifierConcept;

  hasAdjustment(): boolean;
  clearAdjustment(): void;
  getAdjustment(): number;
  setAdjustment(value: number): void;

  hasAction(): boolean;
  clearAction(): void;
  getAction(): ActionMap[keyof ActionMap];
  setAction(value: ActionMap[keyof ActionMap]): void;

  hasClassifierMode(): boolean;
  clearClassifierMode(): void;
  getClassifierMode(): ClassifierModeMap[keyof ClassifierModeMap];
  setClassifierMode(value: ClassifierModeMap[keyof ClassifierModeMap]): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ClassifierCategory.AsObject;
  static toObject(includeInstance: boolean, msg: ClassifierCategory): ClassifierCategory.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: ClassifierCategory, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ClassifierCategory;
  static deserializeBinaryFromReader(message: ClassifierCategory, reader: jspb.BinaryReader): ClassifierCategory;
}

export namespace ClassifierCategory {
  export type AsObject = {
    name: string,
    conceptsList: Array<ClassifierConcept.AsObject>,
    adjustment: number,
    action: ActionMap[keyof ActionMap],
    classifierMode: ClassifierModeMap[keyof ClassifierModeMap],
  }
}

export class ClassifierParameters extends jspb.Message {
  clearCategoriesList(): void;
  getCategoriesList(): Array<ClassifierCategory>;
  setCategoriesList(value: Array<ClassifierCategory>): void;
  addCategories(value?: ClassifierCategory, index?: number): ClassifierCategory;

  clearExceedsList(): void;
  getExceedsList(): Array<ClassifierCategory>;
  setExceedsList(value: Array<ClassifierCategory>): void;
  addExceeds(value?: ClassifierCategory, index?: number): ClassifierCategory;

  hasRealizedAction(): boolean;
  clearRealizedAction(): void;
  getRealizedAction(): ActionMap[keyof ActionMap];
  setRealizedAction(value: ActionMap[keyof ActionMap]): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ClassifierParameters.AsObject;
  static toObject(includeInstance: boolean, msg: ClassifierParameters): ClassifierParameters.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: ClassifierParameters, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ClassifierParameters;
  static deserializeBinaryFromReader(message: ClassifierParameters, reader: jspb.BinaryReader): ClassifierParameters;
}

export namespace ClassifierParameters {
  export type AsObject = {
    categoriesList: Array<ClassifierCategory.AsObject>,
    exceedsList: Array<ClassifierCategory.AsObject>,
    realizedAction: ActionMap[keyof ActionMap],
  }
}

export class AssetParameters extends jspb.Message {
  getAction(): AssetActionMap[keyof AssetActionMap];
  setAction(value: AssetActionMap[keyof AssetActionMap]): void;

  getProject(): string;
  setProject(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): AssetParameters.AsObject;
  static toObject(includeInstance: boolean, msg: AssetParameters): AssetParameters.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: AssetParameters, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): AssetParameters;
  static deserializeBinaryFromReader(message: AssetParameters, reader: jspb.BinaryReader): AssetParameters;
}

export namespace AssetParameters {
  export type AsObject = {
    action: AssetActionMap[keyof AssetActionMap],
    project: string,
  }
}

export class Request extends jspb.Message {
  getEngineId(): string;
  setEngineId(value: string): void;

  getRequestId(): string;
  setRequestId(value: string): void;

  getRequestedType(): ArtifactTypeMap[keyof ArtifactTypeMap];
  setRequestedType(value: ArtifactTypeMap[keyof ArtifactTypeMap]): void;

  clearPromptList(): void;
  getPromptList(): Array<Prompt>;
  setPromptList(value: Array<Prompt>): void;
  addPrompt(value?: Prompt, index?: number): Prompt;

  hasImage(): boolean;
  clearImage(): void;
  getImage(): ImageParameters | undefined;
  setImage(value?: ImageParameters): void;

  hasClassifier(): boolean;
  clearClassifier(): void;
  getClassifier(): ClassifierParameters | undefined;
  setClassifier(value?: ClassifierParameters): void;

  hasAsset(): boolean;
  clearAsset(): void;
  getAsset(): AssetParameters | undefined;
  setAsset(value?: AssetParameters): void;

  hasConditioner(): boolean;
  clearConditioner(): void;
  getConditioner(): ConditionerParameters | undefined;
  setConditioner(value?: ConditionerParameters): void;

  getParamsCase(): Request.ParamsCase;
  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Request.AsObject;
  static toObject(includeInstance: boolean, msg: Request): Request.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Request, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Request;
  static deserializeBinaryFromReader(message: Request, reader: jspb.BinaryReader): Request;
}

export namespace Request {
  export type AsObject = {
    engineId: string,
    requestId: string,
    requestedType: ArtifactTypeMap[keyof ArtifactTypeMap],
    promptList: Array<Prompt.AsObject>,
    image?: ImageParameters.AsObject,
    classifier?: ClassifierParameters.AsObject,
    asset?: AssetParameters.AsObject,
    conditioner?: ConditionerParameters.AsObject,
  }

  export enum ParamsCase {
    PARAMS_NOT_SET = 0,
    IMAGE = 5,
    CLASSIFIER = 7,
    ASSET = 8,
  }
}

export class OnStatus extends jspb.Message {
  clearReasonList(): void;
  getReasonList(): Array<FinishReasonMap[keyof FinishReasonMap]>;
  setReasonList(value: Array<FinishReasonMap[keyof FinishReasonMap]>): void;
  addReason(value: FinishReasonMap[keyof FinishReasonMap], index?: number): FinishReasonMap[keyof FinishReasonMap];

  hasTarget(): boolean;
  clearTarget(): void;
  getTarget(): string;
  setTarget(value: string): void;

  clearActionList(): void;
  getActionList(): Array<StageActionMap[keyof StageActionMap]>;
  setActionList(value: Array<StageActionMap[keyof StageActionMap]>): void;
  addAction(value: StageActionMap[keyof StageActionMap], index?: number): StageActionMap[keyof StageActionMap];

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): OnStatus.AsObject;
  static toObject(includeInstance: boolean, msg: OnStatus): OnStatus.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: OnStatus, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): OnStatus;
  static deserializeBinaryFromReader(message: OnStatus, reader: jspb.BinaryReader): OnStatus;
}

export namespace OnStatus {
  export type AsObject = {
    reasonList: Array<FinishReasonMap[keyof FinishReasonMap]>,
    target: string,
    actionList: Array<StageActionMap[keyof StageActionMap]>,
  }
}

export class Stage extends jspb.Message {
  getId(): string;
  setId(value: string): void;

  hasRequest(): boolean;
  clearRequest(): void;
  getRequest(): Request | undefined;
  setRequest(value?: Request): void;

  clearOnStatusList(): void;
  getOnStatusList(): Array<OnStatus>;
  setOnStatusList(value: Array<OnStatus>): void;
  addOnStatus(value?: OnStatus, index?: number): OnStatus;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Stage.AsObject;
  static toObject(includeInstance: boolean, msg: Stage): Stage.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Stage, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Stage;
  static deserializeBinaryFromReader(message: Stage, reader: jspb.BinaryReader): Stage;
}

export namespace Stage {
  export type AsObject = {
    id: string,
    request?: Request.AsObject,
    onStatusList: Array<OnStatus.AsObject>,
  }
}

export class ChainRequest extends jspb.Message {
  getRequestId(): string;
  setRequestId(value: string): void;

  clearStageList(): void;
  getStageList(): Array<Stage>;
  setStageList(value: Array<Stage>): void;
  addStage(value?: Stage, index?: number): Stage;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ChainRequest.AsObject;
  static toObject(includeInstance: boolean, msg: ChainRequest): ChainRequest.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: ChainRequest, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ChainRequest;
  static deserializeBinaryFromReader(message: ChainRequest, reader: jspb.BinaryReader): ChainRequest;
}

export namespace ChainRequest {
  export type AsObject = {
    requestId: string,
    stageList: Array<Stage.AsObject>,
  }
}

export interface FinishReasonMap {
  NULL: 0;
  LENGTH: 1;
  STOP: 2;
  ERROR: 3;
  FILTER: 4;
}

export const FinishReason: FinishReasonMap;

export interface ArtifactTypeMap {
  ARTIFACT_NONE: 0;
  ARTIFACT_IMAGE: 1;
  ARTIFACT_VIDEO: 2;
  ARTIFACT_TEXT: 3;
  ARTIFACT_TOKENS: 4;
  ARTIFACT_EMBEDDING: 5;
  ARTIFACT_CLASSIFICATIONS: 6;
  ARTIFACT_MASK: 7;
}

export const ArtifactType: ArtifactTypeMap;

export interface DiffusionSamplerMap {
  SAMPLER_DDIM: 0;
  SAMPLER_DDPM: 1;
  SAMPLER_K_EULER: 2;
  SAMPLER_K_EULER_ANCESTRAL: 3;
  SAMPLER_K_HEUN: 4;
  SAMPLER_K_DPM_2: 5;
  SAMPLER_K_DPM_2_ANCESTRAL: 6;
  SAMPLER_K_LMS: 7;
}

export const DiffusionSampler: DiffusionSamplerMap;

export interface UpscalerMap {
  UPSCALER_RGB: 0;
  UPSCALER_GFPGAN: 1;
  UPSCALER_ESRGAN: 2;
}

export const Upscaler: UpscalerMap;

export interface ActionMap {
  ACTION_PASSTHROUGH: 0;
  ACTION_REGENERATE_DUPLICATE: 1;
  ACTION_REGENERATE: 2;
  ACTION_OBFUSCATE_DUPLICATE: 3;
  ACTION_OBFUSCATE: 4;
  ACTION_DISCARD: 5;
}

export const Action: ActionMap;

export interface ClassifierModeMap {
  CLSFR_MODE_ZEROSHOT: 0;
  CLSFR_MODE_MULTICLASS: 1;
}

export const ClassifierMode: ClassifierModeMap;

export interface AssetActionMap {
  ASSET_PUT: 0;
  ASSET_GET: 1;
  ASSET_DELETE: 2;
}

export const AssetAction: AssetActionMap;

export interface StageActionMap {
  STAGE_ACTION_PASS: 0;
  STAGE_ACTION_DISCARD: 1;
  STAGE_ACTION_RETURN: 2;
}

export const StageAction: StageActionMap;

