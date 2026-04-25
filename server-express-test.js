const express = require("express");
const multer = require("multer");
const path = require("path");
const zlib = require("zlib");
const { URL } = require("url");

const ROOT_DIR = __dirname;
const STATIC_DIR = path.join(ROOT_DIR, "static");
const READY_DELAY_MS = 5000;
const LORAS = [
    "https://huggingface.co/lightx2v/Qwen-Image-Lightning/blob/main/Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
    "https://huggingface.co/dx8152/Qwen-Image-Edit-2509-Fusion/blob/main/%E6%BA%B6%E5%9B%BE.safetensors"
];

const app = express();
const upload = multer({ storage: multer.memoryStorage() });
const LORA_FILENAME_TO_URL = buildLoraFilenameMap();

class MockImageStore {
    constructor(readyDelayMs) {
        this.readyDelayMs = readyDelayMs;
        this.items = Object.create(null);
        this.crc32Table = this.createCrc32Table();
        this.fallbackPngBuffer = this.generateSolidPng(500, 500, 0xcc, 0xcc, 0xcc, 255);
    }

    createRequest() {
        const requestId = this.nextRequestId();
        this.items[requestId] = this.fallbackPngBuffer;
        return requestId;
    }

    getPollResponse(requestId) {
        const item = this.items[requestId];
        if (!item) {
            return { status: "pending" };
        }

        if (!this.isReady(requestId)) {
            return { status: "pending" };
        }

        return {
            status: "done",
            url: "/outputs/" + requestId + ".png"
        };
    }

    getOutputByFilename(filename) {
        const match = filename.match(/^(\d+)\.png$/);
        if (!match) {
            return null;
        }

        const requestId = match[1];
        const item = this.items[requestId];
        if (!item) {
            return null;
        }

        if (!this.isReady(requestId)) {
            return null;
        }

        return item;
    }

    nextRequestId() {
        let requestId = String(Date.now());
        while (this.items[requestId]) {
            requestId = String(Number(requestId) + 1);
        }
        return requestId;
    }

    isReady(requestId) {
        return (Date.now() - Number(requestId)) > this.readyDelayMs;
    }

    createCrc32Table() {
        const table = new Uint32Array(256);
        for (let i = 0; i < 256; i += 1) {
            let c = i;
            for (let j = 0; j < 8; j += 1) {
                c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
            }
            table[i] = c >>> 0;
        }
        return table;
    }

    crc32(buffer) {
        let c = 0xffffffff;
        for (let i = 0; i < buffer.length; i += 1) {
            c = this.crc32Table[(c ^ buffer[i]) & 0xff] ^ (c >>> 8);
        }
        return (c ^ 0xffffffff) >>> 0;
    }

    createPngChunk(type, data) {
        const typeBuffer = Buffer.from(type, "ascii");
        const lengthBuffer = Buffer.alloc(4);
        lengthBuffer.writeUInt32BE(data.length, 0);

        const crcBuffer = Buffer.alloc(4);
        crcBuffer.writeUInt32BE(this.crc32(Buffer.concat([typeBuffer, data])), 0);

        return Buffer.concat([lengthBuffer, typeBuffer, data, crcBuffer]);
    }

    generateSolidPng(width, height, red, green, blue, alpha) {
        if (width <= 0 || height <= 0) {
            throw new Error("width and height must be positive");
        }

        const bytesPerPixel = 4;
        const rowLength = 1 + (width * bytesPerPixel);
        const rawData = Buffer.alloc(rowLength * height);

        for (let y = 0; y < height; y += 1) {
            const rowStart = y * rowLength;
            rawData[rowStart] = 0;
            for (let x = 0; x < width; x += 1) {
                const pixelStart = rowStart + 1 + (x * bytesPerPixel);
                rawData[pixelStart] = red;
                rawData[pixelStart + 1] = green;
                rawData[pixelStart + 2] = blue;
                rawData[pixelStart + 3] = alpha;
            }
        }

        const ihdr = Buffer.alloc(13);
        ihdr.writeUInt32BE(width, 0);
        ihdr.writeUInt32BE(height, 4);
        ihdr[8] = 8;
        ihdr[9] = 6;
        ihdr[10] = 0;
        ihdr[11] = 0;
        ihdr[12] = 0;

        const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
        const idat = zlib.deflateSync(rawData);

        return Buffer.concat([
            signature,
            this.createPngChunk("IHDR", ihdr),
            this.createPngChunk("IDAT", idat),
            this.createPngChunk("IEND", Buffer.alloc(0))
        ]);
    }
}

const mockImageStore = new MockImageStore(READY_DELAY_MS);

function parseHuggingFaceUrl(url) {
    try {
        const parsed = new URL(url);
        const pathParts = parsed.pathname.split("/").filter(Boolean);
        if (pathParts.length < 5 || pathParts[2] !== "blob") {
            return null;
        }

        const remaining = pathParts.slice(4);
        const weightName = decodeURIComponent(remaining[remaining.length - 1] || "");
        return { weightName };
    } catch (error) {
        return null;
    }
}

function buildLoraFilenameMap() {
    const mapping = {};
    for (const url of LORAS) {
        const parsed = parseHuggingFaceUrl(url);
        const rawFilename = parsed ? parsed.weightName : url.split("/").pop();
        mapping[rawFilename] = url;
    }
    return mapping;
}

app.get("/api/loras", (req, res) => {
    res.json(Object.keys(LORA_FILENAME_TO_URL));
});

app.post("/api/images", upload.array("images"), (req, res) => {
    const prompt = (req.body.prompt || "").trim();
    if (!prompt) {
        res.status(400).json({ error: "prompt is required" });
        return;
    }

    const requestId = mockImageStore.createRequest();

    res.json({ request_id: requestId });
});

app.get("/api/images/:requestId", (req, res) => {
    const requestId = req.params.requestId;
    if (!/^\d+$/.test(requestId)) {
        res.status(400).json({ error: "invalid request_id" });
        return;
    }

    res.json(mockImageStore.getPollResponse(requestId));
});

app.get("/outputs/:filename", (req, res) => {
    const filename = req.params.filename;
    const buffer = mockImageStore.getOutputByFilename(filename);
    if (!buffer) {
        res.status(404).end();
        return;
    }

    res.setHeader("Content-Type", "image/png");
    res.send(buffer);
});

app.use(express.static(STATIC_DIR));

const portArg = process.argv[2];
const parsedPort = Number.parseInt(portArg, 10);
const port = Number.isInteger(parsedPort) ? parsedPort : 5000;

app.listen(port, "0.0.0.0", () => {
    console.log(`Express test server running at http://localhost:${port}`);
});



// node server-express-test.js 5000