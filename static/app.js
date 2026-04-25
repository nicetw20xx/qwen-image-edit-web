const { createApp } = Vue;

createApp({
  data() {
    return {
      prompt: "",
      inputImages: [],
      availableLoras: [],
      selectedLoras: [],
      historyItems: [],
      previewUrl: null,
      isDragOver: false,
      dragIndex: -1,
      nextImageId: 1,
      nextHistoryId: 1,
      negativePrompt: " ",
      trueCfgScale: 2.0,
      numInferenceSteps: 10,
      numImagesPerPrompt: 1,
    };
  },

  mounted() {
    document.addEventListener("paste", this.handlePaste);
    this.fetchLoras();
  },

  beforeUnmount() {
    document.removeEventListener("paste", this.handlePaste);
  },

  methods: {
    async fetchLoras() {
      try {
        const response = await fetch("/api/loras");
        this.availableLoras = await response.json();
        this.selectedLoras = [...this.availableLoras];
      } catch (e) {
        console.error("Failed to fetch loras:", e);
      }
    },

    decodeName(name) {
      try { return decodeURIComponent(name); } catch { return name; }
    },

    toggleLora(filename) {
      const idx = this.selectedLoras.indexOf(filename);
      if (idx === -1) {
        this.selectedLoras.push(filename);
      } else {
        this.selectedLoras.splice(idx, 1);
      }
    },

    createImageEntry(file) {
      const id = this.nextImageId++;
      return { id, file, objectUrl: URL.createObjectURL(file), name: file.name };
    },

    addFilesToInput(files) {
      for (const file of files) {
        if (file.type.startsWith("image/")) {
          this.inputImages.push(this.createImageEntry(file));
        }
      }
    },

    handleFileDrop(event) {
      this.isDragOver = false;
      this.addFilesToInput(event.dataTransfer.files);
    },

    handleFileSelect(event) {
      this.addFilesToInput(event.target.files);
      event.target.value = "";
    },

    handlePaste(event) {
      const items = event.clipboardData?.items;
      if (!items) return;
      const files = [];
      for (const item of items) {
        if (item.type.startsWith("image/")) {
          const file = item.getAsFile();
          if (file) files.push(file);
        }
      }
      if (files.length) this.addFilesToInput(files);
    },

    removeInputImage(index) {
      const removed = this.inputImages.splice(index, 1)[0];
      URL.revokeObjectURL(removed.objectUrl);
    },

    onThumbDragStart(index, event) {
      this.dragIndex = index;
      event.dataTransfer.effectAllowed = "move";
    },

    onThumbDragOver(index) {
      if (this.dragIndex === -1 || this.dragIndex === index) return;
      const dragged = this.inputImages.splice(this.dragIndex, 1)[0];
      this.inputImages.splice(index, 0, dragged);
      this.dragIndex = index;
    },

    onThumbDrop(index) {
      this.dragIndex = -1;
    },

    onThumbDragEnd() {
      this.dragIndex = -1;
    },

    openPreview(url) {
      this.previewUrl = url;
    },

    addFileToInput(file) {
      this.inputImages.push(this.createImageEntry(file));
    },

    async addToInput(imageUrl) {
      try {
        const response = await fetch(imageUrl);
        const blob = await response.blob();
        const filename = imageUrl.split("/").pop() || "image.png";
        const file = new File([blob], filename, { type: blob.type });
        this.inputImages.push(this.createImageEntry(file));
      } catch (e) {
        console.error("Failed to add image:", e);
      }
    },

    async submitRequest() {
      const prompt = this.prompt.trim();
      if (!prompt) return;
      await this._doRequest({
        prompt,
        negativePrompt: this.negativePrompt,
        loraFilenames: [...this.selectedLoras],
        trueCfgScale: this.trueCfgScale,
        numInferenceSteps: this.numInferenceSteps,
        numImagesPerPrompt: this.numImagesPerPrompt,
        inputImages: this.inputImages.map((img) => ({ objectUrl: img.objectUrl, file: img.file })),
      });
    },

    async resendRequest(item) {
      await this._doRequest({
        prompt: item.prompt,
        negativePrompt: item.negativePrompt,
        loraFilenames: [...item.loraFilenames],
        trueCfgScale: item.trueCfgScale,
        numInferenceSteps: item.numInferenceSteps,
        numImagesPerPrompt: item.numImagesPerPrompt,
        inputImages: item.inputImages.map((img) => ({ objectUrl: img.objectUrl, file: img.file })),
      });
    },

    async _doRequest({ prompt, negativePrompt, loraFilenames, trueCfgScale, numInferenceSteps, numImagesPerPrompt, inputImages }) {
      const formData = new FormData();
      formData.append("prompt", prompt);
      formData.append("negative_prompt", negativePrompt);
      formData.append("true_cfg_scale", trueCfgScale);
      formData.append("num_inference_steps", numInferenceSteps);
      formData.append("num_images_per_prompt", numImagesPerPrompt);
      for (const lora of loraFilenames) formData.append("lora", lora);
      for (const img of inputImages) formData.append("images", img.file);

      const historyEntry = {
        id: this.nextHistoryId++,
        prompt,
        negativePrompt,
        loraFilenames,
        trueCfgScale,
        numInferenceSteps,
        numImagesPerPrompt,
        inputImages,
        status: "generating",
        resultUrls: [],
        resultFiles: [],
        errorMessage: null,
      };

      this.historyItems.unshift(historyEntry);
      const entry = this.historyItems[0];

      try {
        const response = await fetch("/api/images", { method: "POST", body: formData });
        if (!response.ok) {
          const data = await response.json().catch(() => null);
          entry.errorMessage = data?.error || `HTTP ${response.status}`;
          entry.status = "error";
          return;
        }
        const { request_id } = await response.json();
        this.pollResult(entry, request_id);
      } catch (e) {
        entry.errorMessage = e.message || "Network error";
        entry.status = "error";
      }
    },

    pollResult(entry, requestId) {
      const poll = async () => {
        try {
          const res = await fetch(`/api/images/${requestId}`);
          const data = await res.json();

          if (data.status === "done") {
            const resultUrls = Array.isArray(data.images) ? data.images : [];
            entry.resultUrls = resultUrls;
            entry.resultFiles = new Array(resultUrls.length).fill(null);
            entry.status = "done";
            resultUrls.forEach((url, idx) => {
              fetch(url)
                .then((r) => r.blob())
                .then((blob) => {
                  const filename = url.split("/").pop() || `result_${idx}.png`;
                  entry.resultFiles[idx] = new File([blob], filename, { type: blob.type });
                })
                .catch(() => {});
            });
            return;
          }

          if (data.status === "error") {
            entry.errorMessage = data.error || "Inference failed";
            entry.status = "error";
            return;
          }

          setTimeout(poll, 2000);
        } catch (e) {
          entry.errorMessage = e.message || "Polling error";
          entry.status = "error";
        }
      };
      setTimeout(poll, 2000);
    },
  },
}).mount("#app");
