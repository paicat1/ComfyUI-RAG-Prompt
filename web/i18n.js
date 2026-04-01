import { app } from "/scripts/app.js";

const LANGUAGE_SETTING_ID = "EasyRAG.Language";
let LANGUAGE_SETTING_REGISTERED = false;

const TRANSLATIONS = {
  en: {
    "EasyRAG 语言": "EasyRAG Language",
    "跟随 ComfyUI": "Follow ComfyUI",
    "中文": "Chinese",
    "英文": "English",
    "上传文档": "Upload document",
    "文档上传失败: {error}": "Document upload failed: {error}",
    "切换后刷新页面即可生效。": "Refresh the page after switching to apply the change.",
    "EasyRAG - 文档加载": "EasyRAG - Document Loader",
    "EasyRAG - 向量库构建(FAISS)": "EasyRAG - Vector Store Builder (FAISS)",
    "EasyRAG - LM Studio API (高级)": "EasyRAG - LM Studio API (Advanced)",
    "EasyRAG - LM Studio API (简约)": "EasyRAG - LM Studio API (Simple)",
  },
  zh: {
    "EasyRAG Language": "EasyRAG 语言",
    "Follow ComfyUI": "跟随 ComfyUI",
    "Chinese": "中文",
    "English": "英文",
    "Upload document": "上传文档",
    "Document upload failed: {error}": "文档上传失败: {error}",
    "切换后刷新页面即可生效。": "切换后刷新页面即可生效。",
    "EasyRAG - Document Loader": "EasyRAG - 文档加载",
    "EasyRAG - Vector Store Builder (FAISS)": "EasyRAG - 向量库构建(FAISS)",
    "EasyRAG - LM Studio API (Advanced)": "EasyRAG - LM Studio API (高级)",
    "EasyRAG - LM Studio API (Simple)": "EasyRAG - LM Studio API (简约)",
  },
};

function normalizeLanguage(language) {
  const raw = String(language || "").trim().toLowerCase();
  if (!raw || raw === "auto") return "auto";
  if (["zh", "zh-cn", "zh-hans", "cn", "zn"].includes(raw)) return "zh";
  if (["en", "en-us", "en-gb"].includes(raw)) return "en";
  return "en";
}

export function getEasyRAGLanguage() {
  const settingValue = app?.ui?.settings?.getSettingValue?.(LANGUAGE_SETTING_ID) ?? "auto";
  const normalized = normalizeLanguage(settingValue);
  if (normalized !== "auto") return normalized;
  const browserLanguage = document.documentElement.lang || navigator.language || "en";
  const detected = normalizeLanguage(browserLanguage);
  return detected === "auto" ? "zh" : detected;
}

export function t(text, extras) {
  const lang = extras?.language ? normalizeLanguage(extras.language) : getEasyRAGLanguage();
  const mapped = TRANSLATIONS[lang]?.[text] ?? text;
  return extras?.values
    ? mapped.replace(/\{(\w+)\}/g, (_, key) => (key in extras.values ? extras.values[key] : `{${key}}`))
    : mapped;
}

export function localizeNodeTitle(nodeName) {
  const key = {
    EasyRAGDocumentLoader: "EasyRAG - 文档加载",
    EasyRAGVectorStoreBuilder: "EasyRAG - 向量库构建(FAISS)",
    EasyRAGLMStudioChatAdvanced: "EasyRAG - LM Studio API (高级)",
    EasyRAGLMStudioChatSimple: "EasyRAG - LM Studio API (简约)",
  }[nodeName];
  if (!key) return nodeName;
  return t(key);
}

export function registerEasyRAGLanguageSetting() {
  if (LANGUAGE_SETTING_REGISTERED || !app?.ui?.settings?.addSetting) return;
  LANGUAGE_SETTING_REGISTERED = true;

  const existing = app.ui.settings.getSettingValue?.(LANGUAGE_SETTING_ID);
  if (existing === undefined) {
    try {
      app.ui.settings.setSettingValue?.(LANGUAGE_SETTING_ID, "auto");
    } catch {}
  }

  app.ui.settings.addSetting({
    id: LANGUAGE_SETTING_ID,
    name: t("EasyRAG 语言"),
    type: "combo",
    options: [
      { value: "auto", text: t("跟随 ComfyUI") },
      { value: "zh", text: t("中文") },
      { value: "en", text: t("英文") },
    ],
    defaultValue: "auto",
    tooltip: t("切换后刷新页面即可生效。"),
  });
}
