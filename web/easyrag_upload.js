import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { t } from "./i18n.js";

const TARGET_NODE_NAMES = new Set([
  "EasyRAGDocumentLoader",
  "EasyRAGMultiDocumentLoader",
]);

function getComboValues(widget) {
  if (!widget) return [];
  if (Array.isArray(widget.options?.values)) return widget.options.values;
  if (Array.isArray(widget.options)) return widget.options;
  return [];
}

function setComboValues(widget, values) {
  if (!widget) return;
  if (widget.options && Array.isArray(widget.options.values)) {
    widget.options.values = values;
    return;
  }
  if (widget.options && !Array.isArray(widget.options.values)) {
    widget.options.values = values;
    return;
  }
  widget.options = values;
}

app.registerExtension({
  name: "comfyui-easy-rag.document-upload",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!TARGET_NODE_NAMES.has(nodeData.name)) {
      return;
    }

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;

      const hasButton = (this.widgets || []).some((w) => w.name === "upload_document");
      if (hasButton) return r;

      const self = this;
      this.addWidget("button", "upload_document", t("上传文档"), async () => {
        const picker = document.createElement("input");
        picker.type = "file";
        picker.accept = ".txt,.json,.md,.pdf";
        const docListWidget = (self.widgets || []).find((w) => w.name === "documents_text");
        picker.multiple = Boolean(docListWidget);
        picker.style.display = "none";

        picker.onchange = async () => {
          const files = Array.from(picker.files || []);
          if (files.length === 0) return;

          try {
            const docWidget = (self.widgets || []).find((w) => w.name === "document");
            const added = [];

            for (const file of files) {
              const formData = new FormData();
              formData.append("image", file); // ComfyUI 上传接口字段名固定为 image
              formData.append("type", "input");

              const resp = await api.fetchApi("/upload/image", {
                method: "POST",
                body: formData,
              });
              if (!resp.ok) {
                const text = await resp.text();
                throw new Error(text || `HTTP ${resp.status}`);
              }
              const data = await resp.json();
              const savedName = data?.name || file.name;
              added.push(savedName);

              if (docWidget) {
                const values = getComboValues(docWidget).slice();
                if (!values.includes(savedName)) {
                  values.push(savedName);
                  values.sort();
                  setComboValues(docWidget, values);
                }
              }
            }

            if (docWidget && added.length > 0) {
              docWidget.value = added[added.length - 1];
            }
            if (docListWidget && added.length > 0) {
              const existing = String(docListWidget.value || "").trim();
              const parts = existing ? existing.split(/\r?\n/).map((x) => x.trim()).filter(Boolean) : [];
              const merged = [...parts];
              for (const name of added) {
                if (!merged.includes(name)) merged.push(name);
              }
              docListWidget.value = merged.join("\n");
            }
            if (app.graph) {
              app.graph.setDirtyCanvas(true, true);
            }
          } catch (err) {
            console.error("[EasyRAG][上传] 文档上传失败:", err);
            alert(t("Document upload failed: {error}", { values: { error: err?.message || err } }));
          } finally {
            picker.remove();
          }
        };

        document.body.appendChild(picker);
        picker.click();
      });

      return r;
    };
  },
});
