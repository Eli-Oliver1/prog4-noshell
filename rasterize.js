/*
 * Program 4 - textured WebGL renderer
 * Implements texture mapping, Phong lighting, material/texture alpha,
 * painter sorting, and a triangle-splitting BSP transparency pass.
 */

'use strict';

// Use local JSON by default. Override with ?triangles=...&ellipsoids=...
const queryParams = new URLSearchParams(window.location.search);
const INPUT_TRIANGLES_URL = queryParams.get('triangles') || './triangles.json';
const INPUT_ELLIPSOIDS_URL = queryParams.get('ellipsoids') || './ellipsoids.json';

const DEFAULT_EYE = [0.5, 0.5, -0.5];
const DEFAULT_CENTER = [0.5, 0.5, 0.5];
const DEFAULT_UP = [0, 1, 0];
const LIGHT_AMBIENT = [0.65, 0.65, 0.72];
const LIGHT_DIFFUSE = [1.0, 0.96, 0.90];
const LIGHT_SPECULAR = [1.0, 1.0, 1.0];
const LIGHT_POSITION = [-0.5, 1.5, -0.5];
const ROTATE_STEP = Math.PI / 50;
const BSP_EPSILON = 1e-5;
const BSP_MAX_DEPTH = 48;
const BSP_MAX_POLYGONS = 12000;

let gl = null;
let shaderProgram = null;
let locations = {};
let imageContext = null;
let imageCanvas = null;
let webGLCanvas = null;
let viewDelta = 0.01;
let Eye = vec3.clone(DEFAULT_EYE);
let Center = vec3.clone(DEFAULT_CENTER);
let Up = vec3.clone(DEFAULT_UP);
let lightingEnabled = true;
let blendMode = 1; // 0 = replace texture, 1 = modulate texture by lighting
let transparencyMode = 0; // 0 = unsorted, 1 = painter sort, 2 = BSP
let currentScene = { triangles: [], ellipsoids: [] };
let originalScene = null;
let customSceneActive = false;
let textureCache = new Map();
let dummyTexture = null;
let selectedModel = null;
let lastFrameTime = 0;
let stats = { fps: 0, polygons: 0, bspPolygons: 0 };
let sceneGeometryVersion = 0;
let bspCacheVersion = -1;
let bspCacheTree = null;

const VERTEX_SHADER_SOURCE = `
attribute vec3 aPosition;
attribute vec3 aNormal;
attribute vec2 aUV;
uniform mat4 uModel;
uniform mat4 uMVP;
varying vec3 vWorldPosition;
varying vec3 vWorldNormal;
varying vec2 vUV;
void main(void) {
    vec4 worldPosition = uModel * vec4(aPosition, 1.0);
    vWorldPosition = worldPosition.xyz;
    vec4 worldNormal = uModel * vec4(aNormal, 0.0);
    vWorldNormal = normalize(worldNormal.xyz);
    vUV = aUV;
    gl_Position = uMVP * vec4(aPosition, 1.0);
}
`;

const FRAGMENT_SHADER_SOURCE = `
precision mediump float;
uniform vec3 uEyePosition;
uniform vec3 uLightAmbient;
uniform vec3 uLightDiffuse;
uniform vec3 uLightSpecular;
uniform vec3 uLightPosition;
uniform vec3 uAmbient;
uniform vec3 uDiffuse;
uniform vec3 uSpecular;
uniform vec3 uBaseColor;
uniform float uShininess;
uniform float uMaterialAlpha;
uniform bool uUseTexture0;
uniform bool uUseTexture1;
uniform bool uLightingEnabled;
uniform int uBlendMode;
uniform sampler2D uTexture0;
uniform sampler2D uTexture1;
varying vec3 vWorldPosition;
varying vec3 vWorldNormal;
varying vec2 vUV;
void main(void) {
    vec4 texel = uUseTexture0 ? texture2D(uTexture0, vUV) : vec4(uBaseColor, 1.0);
    if (uUseTexture1) {
        vec4 lightMap = texture2D(uTexture1, vUV);
        texel.rgb *= lightMap.rgb;
        texel.a *= lightMap.a;
    }
    float alpha = clamp(texel.a * uMaterialAlpha, 0.0, 1.0);
    if (alpha < 0.015) discard;

    vec3 litColor = uDiffuse * uLightAmbient * uAmbient;
    if (uLightingEnabled) {
        vec3 N = normalize(vWorldNormal);
        vec3 L = normalize(uLightPosition - vWorldPosition);
        vec3 V = normalize(uEyePosition - vWorldPosition);
        float diffuseAmount = max(dot(N, L), 0.0);
        vec3 ambientTerm = uAmbient * uLightAmbient;
        vec3 diffuseTerm = uDiffuse * uLightDiffuse * diffuseAmount;
        vec3 H = normalize(L + V);
        float specularAmount = pow(max(dot(N, H), 0.0), max(uShininess, 1.0));
        vec3 specularTerm = uSpecular * uLightSpecular * specularAmount;
        litColor = ambientTerm + diffuseTerm + specularTerm;
    } else {
        litColor = uBaseColor;
    }

    vec3 resultColor;
    if (uBlendMode == 0) {
        // Replace: show the texture without lighting (or the material color if untextured).
        resultColor = texel.rgb;
    } else {
        // Modulate: the texture scales the Phong-lit material color.
        resultColor = litColor * texel.rgb;
    }
    gl_FragColor = vec4(clamp(resultColor, 0.0, 1.0), alpha);
}
`;

function logError(message, error) {
    console.error(message, error || '');
    setStatus(`${message}${error ? `: ${error.message || error}` : ''}`);
}

function setStatus(message) {
    const status = document.getElementById('status');
    if (status) status.textContent = message;
}

function setStats() {
    const el = document.getElementById('stats');
    if (!el) return;
    const modeNames = ['unsorted', 'painter sorted', 'BSP sorted'];
    const bspInfo = transparencyMode === 2 ? ` · BSP faces ${stats.bspPolygons}` : '';
    el.textContent = `${stats.polygons} source faces · ${modeNames[transparencyMode]}${bspInfo} · ${blendMode === 0 ? 'replace' : 'modulate'} · lighting ${lightingEnabled ? 'on' : 'off'} · ${stats.fps} FPS`;
}

function compileShader(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const message = gl.getShaderInfoLog(shader) || 'Unknown shader compile error';
        gl.deleteShader(shader);
        throw new Error(message);
    }
    return shader;
}

function setupWebGL() {
    imageCanvas = document.getElementById('myImageCanvas');
    webGLCanvas = document.getElementById('myWebGLCanvas');
    imageContext = imageCanvas.getContext('2d');

    gl = webGLCanvas.getContext('webgl', { alpha: true, antialias: true, premultipliedAlpha: false });
    if (!gl) throw new Error('WebGL is unavailable. Try Chrome or Firefox with hardware acceleration enabled.');

    gl.viewport(0, 0, webGLCanvas.width, webGLCanvas.height);
    gl.clearColor(0.015, 0.02, 0.055, 0.0); // keep the 2D sky canvas visible behind WebGL
    gl.clearDepth(1.0);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.disable(gl.CULL_FACE);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    const vertexShader = compileShader(gl.VERTEX_SHADER, VERTEX_SHADER_SOURCE);
    const fragmentShader = compileShader(gl.FRAGMENT_SHADER, FRAGMENT_SHADER_SOURCE);
    shaderProgram = gl.createProgram();
    gl.attachShader(shaderProgram, vertexShader);
    gl.attachShader(shaderProgram, fragmentShader);
    gl.linkProgram(shaderProgram);
    if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(shaderProgram) || 'Shader program link failed');
    }
    gl.useProgram(shaderProgram);

    locations = {
        position: gl.getAttribLocation(shaderProgram, 'aPosition'),
        normal: gl.getAttribLocation(shaderProgram, 'aNormal'),
        uv: gl.getAttribLocation(shaderProgram, 'aUV'),
        model: gl.getUniformLocation(shaderProgram, 'uModel'),
        mvp: gl.getUniformLocation(shaderProgram, 'uMVP'),
        eye: gl.getUniformLocation(shaderProgram, 'uEyePosition'),
        lightAmbient: gl.getUniformLocation(shaderProgram, 'uLightAmbient'),
        lightDiffuse: gl.getUniformLocation(shaderProgram, 'uLightDiffuse'),
        lightSpecular: gl.getUniformLocation(shaderProgram, 'uLightSpecular'),
        lightPosition: gl.getUniformLocation(shaderProgram, 'uLightPosition'),
        ambient: gl.getUniformLocation(shaderProgram, 'uAmbient'),
        diffuse: gl.getUniformLocation(shaderProgram, 'uDiffuse'),
        specular: gl.getUniformLocation(shaderProgram, 'uSpecular'),
        baseColor: gl.getUniformLocation(shaderProgram, 'uBaseColor'),
        shininess: gl.getUniformLocation(shaderProgram, 'uShininess'),
        alpha: gl.getUniformLocation(shaderProgram, 'uMaterialAlpha'),
        useTexture0: gl.getUniformLocation(shaderProgram, 'uUseTexture0'),
        useTexture1: gl.getUniformLocation(shaderProgram, 'uUseTexture1'),
        lighting: gl.getUniformLocation(shaderProgram, 'uLightingEnabled'),
        blendMode: gl.getUniformLocation(shaderProgram, 'uBlendMode'),
        texture0: gl.getUniformLocation(shaderProgram, 'uTexture0'),
        texture1: gl.getUniformLocation(shaderProgram, 'uTexture1')
    };

    gl.enableVertexAttribArray(locations.position);
    gl.enableVertexAttribArray(locations.normal);
    gl.enableVertexAttribArray(locations.uv);
    gl.uniform3fv(locations.lightAmbient, LIGHT_AMBIENT);
    gl.uniform3fv(locations.lightDiffuse, LIGHT_DIFFUSE);
    gl.uniform3fv(locations.lightSpecular, LIGHT_SPECULAR);
    gl.uniform3fv(locations.lightPosition, LIGHT_POSITION);
    gl.uniform1i(locations.texture0, 0);
    gl.uniform1i(locations.texture1, 1);

    dummyTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, dummyTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
        new Uint8Array([255, 255, 255, 255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    loadBackgroundImage('./sky.jpg');
    document.addEventListener('keydown', handleKeyDown);
}

function loadBackgroundImage(url) {
    const image = new Image();
    image.crossOrigin = 'anonymous';
    image.onload = () => {
        imageContext.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
        imageContext.drawImage(image, 0, 0, imageCanvas.width, imageCanvas.height);
    };
    image.onerror = () => console.warn(`Background image could not be loaded: ${url}`);
    image.src = url;
}

async function fetchJSON(url, description, allowMissing) {
    try {
        const response = await fetch(url, { cache: 'no-store' });
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        return await response.json();
    } catch (error) {
        if (allowMissing) {
            console.warn(`Optional ${description} file not loaded (${url}):`, error);
            return [];
        }
        throw new Error(`Could not load ${description} JSON at ${url}. Run this folder from a local web server. ${error.message}`);
    }
}

function safeVec3(value, fallback) {
    if (!Array.isArray(value) || value.length < 3) return fallback.slice();
    return [Number(value[0]) || 0, Number(value[1]) || 0, Number(value[2]) || 0];
}

function normalizeMaterial(source) {
    const material = source || {};
    return {
        ambient: safeVec3(material.ambient, [0.12, 0.12, 0.16]),
        diffuse: safeVec3(material.diffuse, [0.8, 0.8, 0.8]),
        specular: safeVec3(material.specular, [0.25, 0.25, 0.25]),
        n: Math.max(1, Number(material.n ?? material.shininess ?? 16)),
        alpha: Math.max(0, Math.min(1, Number(material.alpha ?? 1))),
        texture: material.texture || material.diffuseTexture || null,
        texture2: material.texture2 || material.lightmap || material.lightMap || material.secondaryTexture || null,
        _texture0: null,
        _texture1: null
    };
}

function computeVertexNormals(vertices, triangles) {
    const normals = vertices.map(() => vec3.create());
    for (const tri of triangles) {
        const a = vertices[tri[0]], b = vertices[tri[1]], c = vertices[tri[2]];
        const ab = vec3.subtract(vec3.create(), b, a);
        const ac = vec3.subtract(vec3.create(), c, a);
        const n = vec3.normalize(vec3.create(), vec3.cross(vec3.create(), ab, ac));
        for (const index of tri) vec3.add(normals[index], normals[index], n);
    }
    return normals.map(n => Array.from(vec3.normalize(n, n)));
}

function makeModel(name, vertices, normals, uvs, triangles, material, type) {
    const cleanVertices = vertices.map(v => safeVec3(v, [0, 0, 0]));
    const cleanTriangles = (triangles || []).map(t => [Number(t[0]), Number(t[1]), Number(t[2])])
        .filter(t => t.every(index => Number.isInteger(index) && index >= 0 && index < cleanVertices.length));
    const cleanNormals = (normals && normals.length === cleanVertices.length)
        ? normals.map(n => safeVec3(n, [0, 0, -1]))
        : computeVertexNormals(cleanVertices, cleanTriangles);
    const cleanUVs = (uvs && uvs.length === cleanVertices.length)
        ? uvs.map(uv => [Number(uv[0]) || 0, Number(uv[1]) || 0])
        : cleanVertices.map(() => [0, 0]);

    const center = vec3.create();
    for (const v of cleanVertices) vec3.add(center, center, v);
    if (cleanVertices.length) vec3.scale(center, center, 1 / cleanVertices.length);

    const flat = [];
    const indices = [];
    for (let i = 0; i < cleanVertices.length; i++) {
        flat.push(...cleanVertices[i], ...cleanNormals[i], ...cleanUVs[i]);
    }
    for (const tri of cleanTriangles) {
        indices.push(tri[0], tri[1], tri[2]);
    }

    const model = {
        name,
        type: type || 'triangle-set',
        vertices: cleanVertices,
        normals: cleanNormals,
        uvs: cleanUVs,
        triangles: cleanTriangles,
        flatVertices: flat,
        indices,
        material: normalizeMaterial(material),
        center,
        translation: vec3.create(),
        xAxis: vec3.fromValues(1, 0, 0),
        yAxis: vec3.fromValues(0, 1, 0),
        on: false,
        vertexBuffer: null,
        indexBuffer: null,
        transparent: false
    };
    model.transparent = model.material.alpha < 0.999;
    return model;
}

function makeEllipsoidMesh(ellipsoid, segments) {
    const vertices = [];
    const normals = [];
    const uvs = [];
    const triangles = [];
    const rings = Math.max(8, Math.floor(segments / 2));
    const around = Math.max(12, segments);
    const rx = Number(ellipsoid.a) || 0.1;
    const ry = Number(ellipsoid.b) || 0.1;
    const rz = Number(ellipsoid.c) || 0.1;
    const cx = Number(ellipsoid.x) || 0.5;
    const cy = Number(ellipsoid.y) || 0.5;
    const cz = Number(ellipsoid.z) || 0.5;

    for (let i = 0; i <= rings; i++) {
        const v = i / rings;
        const theta = v * Math.PI;
        for (let j = 0; j <= around; j++) {
            const u = j / around;
            const phi = u * Math.PI * 2;
            const nx = Math.sin(theta) * Math.cos(phi);
            const ny = Math.cos(theta);
            const nz = Math.sin(theta) * Math.sin(phi);
            vertices.push([cx + rx * nx, cy + ry * ny, cz + rz * nz]);
            normals.push(vec3.normalize(vec3.create(), [nx / rx, ny / ry, nz / rz]).map(Number));
            uvs.push([u, 1 - v]);
        }
    }
    for (let i = 0; i < rings; i++) {
        for (let j = 0; j < around; j++) {
            const a = i * (around + 1) + j;
            const b = a + around + 1;
            triangles.push([a, b, a + 1], [a + 1, b, b + 1]);
        }
    }
    return { vertices, normals, uvs, triangles };
}

function buildScene(triangleData, ellipsoidData) {
    const triangles = (Array.isArray(triangleData) ? triangleData : []).map((set, index) => {
        const material = normalizeMaterial(set.material || {});
        const mesh = makeModel(`Triangle set ${index + 1}`, set.vertices || [], set.normals || null,
            set.uvs || null, set.triangles || [], material, 'triangle-set');
        return mesh;
    });
    const ellipsoids = (Array.isArray(ellipsoidData) ? ellipsoidData : []).map((item, index) => {
        const mesh = makeEllipsoidMesh(item, 32);
        const material = normalizeMaterial(item.material || item);
        return makeModel(`Ellipsoid ${index + 1}`, mesh.vertices, mesh.normals, mesh.uvs,
            mesh.triangles, material, 'ellipsoid');
    });
    return { triangles, ellipsoids };
}

function allModels(scene) {
    return [...scene.triangles, ...scene.ellipsoids];
}

function textureURL(name) {
    if (!name) return null;
    try {
        return new URL(name, new URL(INPUT_TRIANGLES_URL, window.location.href)).href;
    } catch (_) {
        return name;
    }
}

function detectImageTransparency(image) {
    try {
        const canvas = document.createElement('canvas');
        const width = Math.min(96, image.naturalWidth || image.width);
        const height = Math.min(96, image.naturalHeight || image.height);
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext('2d', { willReadFrequently: true });
        context.drawImage(image, 0, 0, width, height);
        const data = context.getImageData(0, 0, width, height).data;
        for (let i = 3; i < data.length; i += 4) if (data[i] < 250) return true;
        return false;
    } catch (error) {
        // If a remote texture cannot be inspected, treating it as transparent is safer.
        console.warn('Could not inspect texture alpha; using transparent render pass.', error);
        return true;
    }
}

function loadTexture(name) {
    const url = textureURL(name);
    if (!url) return Promise.resolve(null);
    if (textureCache.has(url)) return textureCache.get(url).promise;

    const record = { texture: dummyTexture, hasTransparency: false, loaded: false, promise: null };
    record.promise = new Promise(resolve => {
        const image = new Image();
        image.crossOrigin = 'anonymous';
        image.onload = () => {
            const texture = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, texture);
            gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            record.texture = texture;
            record.hasTransparency = detectImageTransparency(image);
            record.loaded = true;
            resolve(record);
        };
        image.onerror = () => {
            console.warn(`Texture failed to load: ${url}`);
            record.loaded = true;
            resolve(record);
        };
        image.src = url;
    });
    textureCache.set(url, record);
    return record.promise;
}

async function attachTextures(scene) {
    const promises = [];
    for (const model of allModels(scene)) {
        const mat = model.material;
        if (mat.texture) {
            promises.push(loadTexture(mat.texture).then(record => {
                mat._texture0 = record;
                if (record && record.hasTransparency) model.transparent = true;
            }));
        }
        if (mat.texture2) {
            promises.push(loadTexture(mat.texture2).then(record => {
                mat._texture1 = record;
                if (record && record.hasTransparency) model.transparent = true;
            }));
        }
    }
    await Promise.all(promises);
    for (const model of allModels(scene)) {
        model.vertexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, model.vertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(model.flatVertices), gl.STATIC_DRAW);
        model.indexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, model.indexBuffer);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(model.indices), gl.STATIC_DRAW);
        model.transparent = model.transparent || model.material.alpha < 0.999;
    }
}

function bindVertexLayout(buffer) {
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    const stride = 8 * Float32Array.BYTES_PER_ELEMENT;
    gl.vertexAttribPointer(locations.position, 3, gl.FLOAT, false, stride, 0);
    gl.vertexAttribPointer(locations.normal, 3, gl.FLOAT, false, stride, 3 * Float32Array.BYTES_PER_ELEMENT);
    gl.vertexAttribPointer(locations.uv, 2, gl.FLOAT, false, stride, 6 * Float32Array.BYTES_PER_ELEMENT);
}

function bindMaterial(material) {
    const mat = material || normalizeMaterial(null);
    const tex0 = mat._texture0 && mat._texture0.texture ? mat._texture0.texture : dummyTexture;
    const tex1 = mat._texture1 && mat._texture1.texture ? mat._texture1.texture : dummyTexture;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, tex1);
    gl.uniform1i(locations.useTexture0, !!mat.texture);
    gl.uniform1i(locations.useTexture1, !!mat.texture2);
    gl.uniform3fv(locations.ambient, mat.ambient);
    gl.uniform3fv(locations.diffuse, mat.diffuse);
    gl.uniform3fv(locations.specular, mat.specular);
    gl.uniform3fv(locations.baseColor, mat.diffuse);
    gl.uniform1f(locations.shininess, mat.n);
    gl.uniform1f(locations.alpha, mat.alpha);
    gl.uniform1i(locations.lighting, lightingEnabled);
    gl.uniform1i(locations.blendMode, blendMode);
}

function makeModelTransform(model) {
    const center = model.center;
    const m = mat4.create();
    const temp = mat4.create();
    const negCenter = vec3.negate(vec3.create(), center);
    const zAxis = vec3.normalize(vec3.create(), vec3.cross(vec3.create(), model.xAxis, model.yAxis));
    const rotation = mat4.create();
    mat4.set(rotation,
        model.xAxis[0], model.yAxis[0], zAxis[0], 0,
        model.xAxis[1], model.yAxis[1], zAxis[1], 0,
        model.xAxis[2], model.yAxis[2], zAxis[2], 0,
        0, 0, 0, 1);
    mat4.fromTranslation(m, negCenter);
    if (model.on) mat4.multiply(m, mat4.fromScaling(temp, [1.12, 1.12, 1.12]), m);
    mat4.multiply(m, rotation, m);
    mat4.multiply(m, mat4.fromTranslation(temp, center), m);
    mat4.multiply(m, mat4.fromTranslation(temp, model.translation), m);
    return m;
}

function computeViewProjection() {
    const projection = mat4.create();
    const view = mat4.create();
    const pv = mat4.create();
    const fieldOfView = customSceneActive ? (68 * Math.PI / 180) : (Math.PI / 2);
    mat4.perspective(projection, fieldOfView, webGLCanvas.width / webGLCanvas.height, 0.05, 30);
    mat4.lookAt(view, Eye, Center, Up);
    mat4.multiply(pv, projection, view);
    const forward = vec3.normalize(vec3.create(), vec3.subtract(vec3.create(), Center, Eye));
    gl.uniform3fv(locations.eye, Eye);
    return { pv, forward };
}

function transformedTriangle(model, triangleIndex, modelMatrix, forward) {
    const tri = model.triangles[triangleIndex];
    if (!tri) return null;
    const vertices = [];
    const m3 = mat3.create();
    mat3.fromMat4(m3, modelMatrix);
    for (const index of tri) {
        const position = vec3.transformMat4(vec3.create(), model.vertices[index], modelMatrix);
        const normal = vec3.normalize(vec3.create(), vec3.transformMat3(vec3.create(), model.normals[index], m3));
        const uv = model.uvs[index] || [0, 0];
        vertices.push({ position: Array.from(position), normal: Array.from(normal), uv: uv.slice() });
    }
    const center = vec3.create();
    for (const vertex of vertices) vec3.add(center, center, vertex.position);
    vec3.scale(center, center, 1 / 3);
    const depth = vec3.dot(vec3.subtract(vec3.create(), center, Eye), forward);
    return { vertices, material: model.material, model, depth };
}

function drawOpaqueModel(model, pv) {
    const modelMatrix = makeModelTransform(model);
    const mvp = mat4.multiply(mat4.create(), pv, modelMatrix);
    gl.uniformMatrix4fv(locations.model, false, modelMatrix);
    gl.uniformMatrix4fv(locations.mvp, false, mvp);
    bindMaterial(model.material);
    bindVertexLayout(model.vertexBuffer);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, model.indexBuffer);
    gl.drawElements(gl.TRIANGLES, model.indices.length, gl.UNSIGNED_SHORT, 0);
}

function drawUnsortedTransparentModel(model, pv) {
    const modelMatrix = makeModelTransform(model);
    const mvp = mat4.multiply(mat4.create(), pv, modelMatrix);
    gl.uniformMatrix4fv(locations.model, false, modelMatrix);
    gl.uniformMatrix4fv(locations.mvp, false, mvp);
    bindMaterial(model.material);
    bindVertexLayout(model.vertexBuffer);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, model.indexBuffer);
    gl.drawElements(gl.TRIANGLES, model.indices.length, gl.UNSIGNED_SHORT, 0);
}

function drawPainterTriangle(entry, pv) {
    const model = entry.model;
    const modelMatrix = makeModelTransform(model);
    const mvp = mat4.multiply(mat4.create(), pv, modelMatrix);
    gl.uniformMatrix4fv(locations.model, false, modelMatrix);
    gl.uniformMatrix4fv(locations.mvp, false, mvp);
    bindMaterial(model.material);
    bindVertexLayout(model.vertexBuffer);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, model.indexBuffer);
    gl.drawElements(gl.TRIANGLES, 3, gl.UNSIGNED_SHORT, entry.triangleIndex * 3 * Uint16Array.BYTES_PER_ELEMENT);
}

function planeForPolygon(poly) {
    if (!poly || poly.length < 3) return null;
    const a = poly[0].position, b = poly[1].position, c = poly[2].position;
    const ab = vec3.subtract(vec3.create(), b, a);
    const ac = vec3.subtract(vec3.create(), c, a);
    const normal = vec3.normalize(vec3.create(), vec3.cross(vec3.create(), ab, ac));
    if (vec3.squaredLength(normal) < 1e-12) return null;
    return { normal: Array.from(normal), offset: vec3.dot(normal, a) };
}

function interpolateVertex(a, b, t) {
    const mix = (x, y) => x + (y - x) * t;
    return {
        position: [mix(a.position[0], b.position[0]), mix(a.position[1], b.position[1]), mix(a.position[2], b.position[2])],
        normal: vec3.normalize(vec3.create(), [mix(a.normal[0], b.normal[0]), mix(a.normal[1], b.normal[1]), mix(a.normal[2], b.normal[2])]).map(Number),
        uv: [mix(a.uv[0], b.uv[0]), mix(a.uv[1], b.uv[1])]
    };
}

function clipPolygon(poly, plane, keepFront) {
    const output = [];
    for (let i = 0; i < poly.length; i++) {
        const current = poly[i];
        const next = poly[(i + 1) % poly.length];
        const dCurrent = vec3.dot(plane.normal, current.position) - plane.offset;
        const dNext = vec3.dot(plane.normal, next.position) - plane.offset;
        const currentInside = keepFront ? dCurrent >= -BSP_EPSILON : dCurrent <= BSP_EPSILON;
        const nextInside = keepFront ? dNext >= -BSP_EPSILON : dNext <= BSP_EPSILON;
        if (currentInside) output.push(current);
        if (currentInside !== nextInside && Math.abs(dCurrent - dNext) > 1e-12) {
            const t = dCurrent / (dCurrent - dNext);
            output.push(interpolateVertex(current, next, t));
        }
    }
    // Remove adjacent duplicates introduced by points exactly on the splitter.
    const compact = [];
    for (const vertex of output) {
        if (!compact.length || vec3.distance(compact[compact.length - 1].position, vertex.position) > BSP_EPSILON) compact.push(vertex);
    }
    if (compact.length > 2 && vec3.distance(compact[0].position, compact[compact.length - 1].position) <= BSP_EPSILON) compact.pop();
    return compact;
}

function triangulatePolygon(poly, material, model) {
    const result = [];
    if (!poly || poly.length < 3) return result;
    for (let i = 1; i < poly.length - 1; i++) {
        result.push({ vertices: [poly[0], poly[i], poly[i + 1]], material, model, depth: 0 });
    }
    return result;
}

function evaluateSplitter(candidate, polygons) {
    const plane = planeForPolygon(candidate.vertices);
    if (!plane) return Infinity;
    let front = 0, back = 0, spanning = 0;
    for (const polygon of polygons) {
        let hasFront = false, hasBack = false;
        for (const vertex of polygon.vertices) {
            const distance = vec3.dot(plane.normal, vertex.position) - plane.offset;
            if (distance > BSP_EPSILON) hasFront = true;
            else if (distance < -BSP_EPSILON) hasBack = true;
        }
        if (hasFront && hasBack) spanning++;
        else if (hasFront) front++;
        else if (hasBack) back++;
    }
    return spanning * 10 + Math.abs(front - back) * 0.35;
}

function buildBSP(polygons, depth, counter) {
    if (!polygons.length) return null;
    if (depth >= BSP_MAX_DEPTH || counter.count > BSP_MAX_POLYGONS) {
        return { plane: null, coplanar: [], front: null, back: null, fallback: polygons };
    }

    const sampleCount = Math.min(12, polygons.length);
    let splitter = polygons[0];
    let bestScore = Infinity;
    for (let i = 0; i < sampleCount; i++) {
        const candidate = polygons[Math.floor(i * polygons.length / sampleCount)];
        const score = evaluateSplitter(candidate, polygons);
        if (score < bestScore) { bestScore = score; splitter = candidate; }
    }
    const plane = planeForPolygon(splitter.vertices);
    if (!plane) {
        const rest = polygons.slice(1);
        return { plane: null, coplanar: [], front: null, back: null, fallback: polygons };
    }

    const node = { plane, coplanar: [], frontPolygons: [], backPolygons: [], front: null, back: null, fallback: null };
    for (const polygon of polygons) {
        let hasFront = false, hasBack = false;
        const distances = polygon.vertices.map(v => vec3.dot(plane.normal, v.position) - plane.offset);
        for (const d of distances) {
            if (d > BSP_EPSILON) hasFront = true;
            else if (d < -BSP_EPSILON) hasBack = true;
        }
        if (!hasFront && !hasBack) {
            node.coplanar.push(polygon);
        } else if (!hasBack) {
            node.frontPolygons.push(polygon);
        } else if (!hasFront) {
            node.backPolygons.push(polygon);
        } else {
            const frontPoly = clipPolygon(polygon.vertices, plane, true);
            const backPoly = clipPolygon(polygon.vertices, plane, false);
            const frontPieces = triangulatePolygon(frontPoly, polygon.material, polygon.model);
            const backPieces = triangulatePolygon(backPoly, polygon.material, polygon.model);
            node.frontPolygons.push(...frontPieces);
            node.backPolygons.push(...backPieces);
            counter.count += frontPieces.length + backPieces.length;
        }
    }
    if (counter.count > BSP_MAX_POLYGONS) {
        node.fallback = [...node.coplanar, ...node.frontPolygons, ...node.backPolygons];
        node.frontPolygons = [];
        node.backPolygons = [];
        return node;
    }
    node.front = buildBSP(node.frontPolygons, depth + 1, counter);
    node.back = buildBSP(node.backPolygons, depth + 1, counter);
    node.frontPolygons = null;
    node.backPolygons = null;
    return node;
}

function polygonDepth(poly, forward) {
    const center = vec3.create();
    for (const v of poly.vertices) vec3.add(center, center, v.position);
    vec3.scale(center, center, 1 / poly.vertices.length);
    return vec3.dot(vec3.subtract(vec3.create(), center, Eye), forward);
}

function extractBSP(node, ordered, forward) {
    if (!node) return;
    if (node.fallback) {
        node.fallback.sort((a, b) => polygonDepth(b, forward) - polygonDepth(a, forward));
        ordered.push(...node.fallback);
        return;
    }
    if (!node.plane) {
        if (node.coplanar) ordered.push(...node.coplanar);
        return;
    }
    const eyeSide = vec3.dot(node.plane.normal, Eye) - node.plane.offset;
    if (eyeSide >= 0) {
        extractBSP(node.back, ordered, forward);
        if (node.coplanar) ordered.push(...node.coplanar);
        extractBSP(node.front, ordered, forward);
    } else {
        extractBSP(node.front, ordered, forward);
        if (node.coplanar) ordered.push(...node.coplanar);
        extractBSP(node.back, ordered, forward);
    }
}

function drawWorldPolygon(poly, pv) {
    const data = [];
    for (const vertex of poly.vertices) data.push(...vertex.position, ...vertex.normal, ...vertex.uv);
    gl.uniformMatrix4fv(locations.model, false, mat4.create());
    gl.uniformMatrix4fv(locations.mvp, false, pv);
    bindMaterial(poly.material);
    gl.bindBuffer(gl.ARRAY_BUFFER, drawWorldPolygon.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.DYNAMIC_DRAW);
    bindVertexLayout(drawWorldPolygon.buffer);
    gl.drawArrays(gl.TRIANGLE_FAN, 0, poly.vertices.length);
}

function renderFrame(timestamp) {
    if (!gl) return;
    window.requestAnimationFrame(renderFrame);
    if (lastFrameTime) {
        const elapsed = timestamp - lastFrameTime;
        if (elapsed > 0) stats.fps = Math.round(1000 / elapsed);
    }
    lastFrameTime = timestamp;

    gl.viewport(0, 0, webGLCanvas.width, webGLCanvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(shaderProgram);
    const { pv, forward } = computeViewProjection();
    const models = allModels(currentScene);
    stats.polygons = models.reduce((sum, model) => sum + model.triangles.length, 0);

    // Pass 1: opaque geometry writes depth normally.
    gl.disable(gl.BLEND);
    gl.depthMask(true);
    for (const model of models) {
        if (!model.transparent) drawOpaqueModel(model, pv);
    }

    // Pass 2: transparent geometry tests against opaque depth but never writes depth.
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    const transparentModels = models.filter(model => model.transparent);

    if (transparencyMode === 0) {
        // Part 4 baseline: preserve input order; each model is drawn in one call.
        for (const model of transparentModels) drawUnsortedTransparentModel(model, pv);
        stats.bspPolygons = 0;
    } else if (transparencyMode === 1) {
        // Part 6: far-to-near triangle painter sort, one draw call per triangle.
        const triangles = [];
        for (const model of transparentModels) {
            const modelMatrix = makeModelTransform(model);
            for (let i = 0; i < model.triangles.length; i++) {
                const entry = transformedTriangle(model, i, modelMatrix, forward);
                if (entry) { entry.triangleIndex = i; triangles.push(entry); }
            }
        }
        triangles.sort((a, b) => b.depth - a.depth);
        for (const entry of triangles) drawPainterTriangle(entry, pv);
        stats.bspPolygons = 0;
    } else {
        // Part 7: cache the geometry partition, but extract a fresh view order every frame.
        // The version changes whenever a model is moved, rotated, selected, or a scene is replaced.
        if (bspCacheVersion !== sceneGeometryVersion) {
            const polygons = [];
            for (const model of transparentModels) {
                const modelMatrix = makeModelTransform(model);
                for (let i = 0; i < model.triangles.length; i++) {
                    const poly = transformedTriangle(model, i, modelMatrix, forward);
                    if (poly) polygons.push(poly);
                }
            }
            const counter = { count: polygons.length };
            bspCacheTree = buildBSP(polygons, 0, counter);
            bspCacheVersion = sceneGeometryVersion;
        }
        const ordered = [];
        extractBSP(bspCacheTree, ordered, forward);
        for (const poly of ordered) drawWorldPolygon(poly, pv);
        stats.bspPolygons = ordered.length;
    }

    gl.depthMask(true);
    gl.disable(gl.BLEND);
    setStats();
}
drawWorldPolygon.buffer = null;

function cameraVectors() {
    const look = vec3.normalize(vec3.create(), vec3.subtract(vec3.create(), Center, Eye));
    const right = vec3.normalize(vec3.create(), vec3.cross(vec3.create(), look, Up));
    return { look, right };
}

function translateSelected(offset) {
    if (selectedModel) {
        const model = selectedModel.model;
        vec3.add(model.translation, model.translation, offset);
        sceneGeometryVersion++;
    }
}

function rotateSelected(axis, direction) {
    if (!selectedModel) return;
    const model = selectedModel.model;
    const rotation = mat4.create();
    mat4.fromRotation(rotation, direction * ROTATE_STEP, axis);
    vec3.transformMat4(model.xAxis, model.xAxis, rotation);
    vec3.transformMat4(model.yAxis, model.yAxis, rotation);
    sceneGeometryVersion++;
}

function selectModel(kind, index) {
    if (selectedModel) selectedModel.model.on = false;
    const list = kind === 'triangle' ? currentScene.triangles : currentScene.ellipsoids;
    if (!list.length) { selectedModel = null; sceneGeometryVersion++; return; }
    const wrapped = ((index % list.length) + list.length) % list.length;
    selectedModel = { kind, index: wrapped, model: list[wrapped] };
    selectedModel.model.on = true;
    sceneGeometryVersion++;
    setStatus(`Selected ${selectedModel.model.name}. Use K/; L/O I/P to move; hold Shift to rotate.`);
}

function resetModels() {
    for (const model of allModels(currentScene)) {
        vec3.set(model.translation, 0, 0, 0);
        vec3.set(model.xAxis, 1, 0, 0);
        vec3.set(model.yAxis, 0, 1, 0);
        model.on = false;
    }
    selectedModel = null;
    sceneGeometryVersion++;
}

function handleKeyDown(event) {
    const { look, right } = cameraVectors();
    const temp = vec3.create();
    const shifted = event.shiftKey;
    const key = event.key;
    let handled = true;
    switch (event.code) {
        case 'ArrowRight': selectModel('triangle', (selectedModel && selectedModel.kind === 'triangle' ? selectedModel.index + 1 : 0)); break;
        case 'ArrowLeft': selectModel('triangle', (selectedModel && selectedModel.kind === 'triangle' ? selectedModel.index - 1 : currentScene.triangles.length - 1)); break;
        case 'ArrowUp': selectModel('ellipsoid', (selectedModel && selectedModel.kind === 'ellipsoid' ? selectedModel.index + 1 : 0)); break;
        case 'ArrowDown': selectModel('ellipsoid', (selectedModel && selectedModel.kind === 'ellipsoid' ? selectedModel.index - 1 : currentScene.ellipsoids.length - 1)); break;
        case 'Space':
            if (selectedModel) { selectedModel.model.on = false; sceneGeometryVersion++; }
            selectedModel = null;
            setStatus('Selection cleared.');
            break;
        case 'KeyA':
            vec3.scale(temp, right, viewDelta); vec3.add(Center, Center, temp);
            if (!shifted) vec3.add(Eye, Eye, temp);
            break;
        case 'KeyD':
            vec3.scale(temp, right, -viewDelta); vec3.add(Center, Center, temp);
            if (!shifted) vec3.add(Eye, Eye, temp);
            break;
        case 'KeyS':
            if (shifted) {
                vec3.scale(temp, Up, viewDelta); vec3.add(Center, Center, temp);
                vec3.cross(Up, right, look); vec3.normalize(Up, Up);
            } else {
                vec3.scale(temp, look, -viewDelta); vec3.add(Eye, Eye, temp); vec3.add(Center, Center, temp);
            }
            break;
        case 'KeyW':
            if (shifted) {
                vec3.scale(temp, Up, -viewDelta); vec3.add(Center, Center, temp);
                vec3.cross(Up, right, look); vec3.normalize(Up, Up);
            } else {
                vec3.scale(temp, look, viewDelta); vec3.add(Eye, Eye, temp); vec3.add(Center, Center, temp);
            }
            break;
        case 'KeyQ':
            if (shifted) { vec3.scale(temp, right, -viewDelta); vec3.add(Up, Up, temp); vec3.normalize(Up, Up); }
            else { vec3.scale(temp, Up, viewDelta); vec3.add(Eye, Eye, temp); vec3.add(Center, Center, temp); }
            break;
        case 'KeyE':
            if (shifted) { vec3.scale(temp, right, viewDelta); vec3.add(Up, Up, temp); vec3.normalize(Up, Up); }
            else { vec3.scale(temp, Up, -viewDelta); vec3.add(Eye, Eye, temp); vec3.add(Center, Center, temp); }
            break;
        case 'Escape':
            Eye = vec3.clone(DEFAULT_EYE); Center = vec3.clone(DEFAULT_CENTER); Up = vec3.clone(DEFAULT_UP);
            break;
        case 'KeyK':
            if (shifted) rotateSelected(Up, -1); else translateSelected(vec3.scale(temp, right, viewDelta));
            break;
        case 'Semicolon':
            if (shifted) rotateSelected(Up, 1); else translateSelected(vec3.scale(temp, right, -viewDelta));
            break;
        case 'KeyL':
            if (shifted) rotateSelected(right, 1); else translateSelected(vec3.scale(temp, look, -viewDelta));
            break;
        case 'KeyO':
            if (shifted) rotateSelected(right, -1); else translateSelected(vec3.scale(temp, look, viewDelta));
            break;
        case 'KeyI':
            if (shifted) rotateSelected(look, 1); else translateSelected(vec3.scale(temp, Up, viewDelta));
            break;
        case 'KeyP':
            if (shifted) rotateSelected(look, -1); else translateSelected(vec3.scale(temp, Up, -viewDelta));
            break;
        case 'Backspace': resetModels(); break;
        case 'KeyB':
            blendMode = 1 - blendMode;
            setStatus(`Texture blend mode: ${blendMode === 0 ? 'replace' : 'modulate'}.`);
            break;
        case 'KeyN':
            lightingEnabled = !lightingEnabled;
            setStatus(`Lighting ${lightingEnabled ? 'enabled' : 'disabled'}.`);
            break;
        case 'Comma':
            transparencyMode = (transparencyMode + 2) % 3;
            setStatus(`Transparency: ${['unsorted', 'painter sorted', 'BSP sorted'][transparencyMode]}.`);
            break;
        case 'Period':
            transparencyMode = (transparencyMode + 1) % 3;
            setStatus(`Transparency: ${['unsorted', 'painter sorted', 'BSP sorted'][transparencyMode]}.`);
            break;
        case 'Digit1':
            if (shifted) toggleInterestingScene(); else handled = false;
            break;
        case 'Digit0':
            if (!shifted) restoreOriginalScene(); else handled = false;
            break;
        default:
            if (key === '!') toggleInterestingScene();
            else handled = false;
    }
    if (handled) event.preventDefault();
}

function makeQuad(name, corners, material, uvs) {
    const defaultUVs = [[0, 0], [0, 1], [1, 1], [1, 0]];
    return makeModel(name, corners, corners.map(() => [0, 0, -1]), uvs || defaultUVs,
        [[0, 1, 2], [0, 2, 3]], material, 'triangle-set');
}

function makeRing(name, center, innerRadius, outerRadius, zWave, material, segments) {
    const vertices = [], normals = [], uvs = [], triangles = [];
    const count = segments || 96;
    for (let i = 0; i < count; i++) {
        const a0 = i / count * Math.PI * 2;
        const a1 = (i + 1) / count * Math.PI * 2;
        const points = [
            [center[0] + Math.cos(a0) * innerRadius, center[1] + Math.sin(a0) * innerRadius * 0.52, center[2] + Math.sin(a0 * 2) * zWave],
            [center[0] + Math.cos(a0) * outerRadius, center[1] + Math.sin(a0) * outerRadius * 0.52, center[2] + Math.sin(a0 * 2) * zWave],
            [center[0] + Math.cos(a1) * outerRadius, center[1] + Math.sin(a1) * outerRadius * 0.52, center[2] + Math.sin(a1 * 2) * zWave],
            [center[0] + Math.cos(a1) * innerRadius, center[1] + Math.sin(a1) * innerRadius * 0.52, center[2] + Math.sin(a1 * 2) * zWave]
        ];
        const base = vertices.length;
        vertices.push(...points);
        normals.push(...points.map(() => [0, 0, -1]));
        uvs.push([i / count, 0], [i / count, 1], [(i + 1) / count, 1], [(i + 1) / count, 0]);
        triangles.push([base, base + 1, base + 2], [base, base + 2, base + 3]);
    }
    return makeModel(name, vertices, normals, uvs, triangles, material, 'triangle-set');
}

function buildInterestingScene() {
    const triangles = [];
    const ellipsoids = [];
    const materials = {
        ring: { ambient: [0.28, 0.28, 0.32], diffuse: [0.8, 0.9, 1], specular: [0.7, 0.8, 1], n: 36, alpha: 0.78, texture: 'LightMap.png', texture2: 'lightmap.gif' },
        leaf: { ambient: [0.25, 0.3, 0.22], diffuse: [0.85, 1, 0.8], specular: [0.15, 0.25, 0.18], n: 12, alpha: 0.96, texture: 'tree.png' },
        glass: { ambient: [0.28, 0.36, 0.5], diffuse: [0.55, 0.8, 1], specular: [1, 1, 1], n: 64, alpha: 0.34, texture: 'glass.gif' },
        crystal: { ambient: [0.18, 0.22, 0.4], diffuse: [0.8, 0.9, 1], specular: [0.9, 0.95, 1], n: 48, alpha: 0.64, texture: 'abe.png' },
        star: { ambient: [0.2, 0.2, 0.34], diffuse: [1, 0.9, 0.75], specular: [0.8, 0.8, 1], n: 24, alpha: 0.93, texture: 'stars.jpg' }
    };

    // A glowing, textured orbital ring around the central planet.
    triangles.push(makeRing('Luminous orbital ring', [0.5, 0.53, 0.71], 0.205, 0.285, 0.045, materials.ring, 112));

    // Layered crystal sails and botanical cut-outs at distinct depths.
    triangles.push(makeQuad('Crystal sail left', [[0.13, 0.30, 0.73], [0.28, 0.83, 0.75], [0.42, 0.31, 0.72], [0.30, 0.22, 0.71]], materials.crystal,
        [[0, 0], [0.5, 1], [1, 0], [0.5, 0.15]]));
    triangles.push(makeQuad('Crystal sail right', [[0.59, 0.23, 0.66], [0.73, 0.82, 0.69], [0.90, 0.31, 0.67], [0.76, 0.17, 0.66]], { ...materials.crystal, texture: 'abe.png', alpha: 0.48 },
        [[0, 0], [0.5, 1], [1, 0], [0.5, 0.1]]));
    triangles.push(makeQuad('Botanical veil', [[0.05, 0.17, 0.48], [0.18, 0.72, 0.50], [0.44, 0.64, 0.49], [0.29, 0.13, 0.48]], materials.leaf,
        [[0, 1], [0.25, 0], [1, 0], [0.75, 1]]));
    triangles.push(makeQuad('Floating leaf canopy', [[0.55, 0.69, 0.53], [0.68, 0.95, 0.51], [0.94, 0.74, 0.52], [0.77, 0.58, 0.54]], { ...materials.leaf, texture: 'leaf.small.png', alpha: 0.9 },
        [[0, 1], [0.5, 0], [1, 1], [0.5, 0.8]]));

    // A dark tiled plinth below the floating orbs.
    triangles.push(makeQuad('Obsidian plinth', [[0.12, 0.16, 0.90], [0.88, 0.16, 0.90], [0.88, 0.24, 0.90], [0.12, 0.24, 0.90]],
        { ambient: [0.4, 0.4, 0.45], diffuse: [0.55, 0.65, 0.8], specular: [0.4, 0.5, 0.7], n: 20, alpha: 1, texture: 'rocktile.jpg' },
        [[0, 0], [3, 0], [3, 0.4], [0, 0.4]]));

    // Additional translucent fragments provide visible overlap for the sorting modes.
    triangles.push(makeQuad('Amber glass shard', [[0.40, 0.33, 0.42], [0.49, 0.77, 0.44], [0.57, 0.36, 0.43], [0.49, 0.29, 0.41]],
        { ambient: [0.35, 0.19, 0.08], diffuse: [1, 0.65, 0.25], specular: [1, 0.7, 0.35], n: 42, alpha: 0.45, texture: 'retro.jpg' }));
    triangles.push(makeQuad('Violet glass shard', [[0.32, 0.35, 0.37], [0.49, 0.69, 0.38], [0.68, 0.35, 0.39], [0.49, 0.25, 0.37]],
        { ambient: [0.2, 0.12, 0.35], diffuse: [0.65, 0.48, 1], specular: [0.7, 0.65, 1], n: 32, alpha: 0.38, texture: 'lightmap.gif' }));

    const planet = makeEllipsoidMesh({ x: 0.5, y: 0.52, z: 0.78, a: 0.17, b: 0.17, c: 0.075 }, 40);
    ellipsoids.push(makeModel('Aurelia - ocean planet', planet.vertices, planet.normals, planet.uvs, planet.triangles,
        { ambient: [0.16, 0.2, 0.26], diffuse: [0.95, 0.95, 1], specular: [0.9, 0.95, 1], n: 42, alpha: 1, texture: 'earth.png' }, 'ellipsoid'));

    const moon = makeEllipsoidMesh({ x: 0.25, y: 0.73, z: 0.67, a: 0.065, b: 0.065, c: 0.045 }, 28);
    ellipsoids.push(makeModel('Pearl moon', moon.vertices, moon.normals, moon.uvs, moon.triangles,
        { ambient: [0.2, 0.2, 0.28], diffuse: [0.95, 0.92, 1], specular: [1, 1, 1], n: 40, alpha: 0.92, texture: 'stars.jpg' }, 'ellipsoid'));

    const glassOrb = makeEllipsoidMesh({ x: 0.77, y: 0.72, z: 0.57, a: 0.105, b: 0.105, c: 0.065 }, 28);
    ellipsoids.push(makeModel('Glass seed', glassOrb.vertices, glassOrb.normals, glassOrb.uvs, glassOrb.triangles, materials.glass, 'ellipsoid'));

    const seed = makeEllipsoidMesh({ x: 0.73, y: 0.33, z: 0.47, a: 0.058, b: 0.085, c: 0.04 }, 24);
    ellipsoids.push(makeModel('Amber seed', seed.vertices, seed.normals, seed.uvs, seed.triangles,
        { ambient: [0.3, 0.15, 0.04], diffuse: [1, 0.62, 0.22], specular: [0.95, 0.7, 0.35], n: 28, alpha: 0.78, texture: 'billie.jpg' }, 'ellipsoid'));

    return { triangles, ellipsoids };
}

async function activateScene(scene, label) {
    // Build textures and GPU buffers before exposing the new scene to the animation loop.
    await attachTextures(scene);
    currentScene = scene;
    selectedModel = null;
    sceneGeometryVersion++;
    setStatus(label);
}

async function toggleInterestingScene() {
    if (!customSceneActive) {
        if (!originalScene) return;
        const scene = buildInterestingScene();
        Eye = vec3.clone(DEFAULT_EYE); Center = vec3.clone(DEFAULT_CENTER); Up = vec3.clone(DEFAULT_UP);
        await activateScene(scene, 'Part 8: Interdimensional Garden. Press 0 to return to the supplied input scene.');
        customSceneActive = true;
    } else {
        restoreOriginalScene();
    }
}

function restoreOriginalScene() {
    if (!originalScene) return;
    customSceneActive = false;
    currentScene = originalScene;
    selectedModel = null;
    sceneGeometryVersion++;
    setStatus('Supplied input scene restored. Press ! for the Part 8 artwork.');
}

async function main() {
    try {
        setupWebGL();
        drawWorldPolygon.buffer = gl.createBuffer();
        setStatus('Loading triangle and ellipsoid JSON…');
        const [triangleData, ellipsoidData] = await Promise.all([
            fetchJSON(INPUT_TRIANGLES_URL, 'triangles', false),
            fetchJSON(INPUT_ELLIPSOIDS_URL, 'ellipsoids', true)
        ]);
        originalScene = buildScene(triangleData, ellipsoidData);
        currentScene = originalScene;
        await attachTextures(currentScene);
        sceneGeometryVersion++;

        const all = allModels(currentScene);
        let min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
        for (const model of all) for (const vertex of model.vertices) {
            for (let axis = 0; axis < 3; axis++) {
                min[axis] = Math.min(min[axis], vertex[axis]);
                max[axis] = Math.max(max[axis], vertex[axis]);
            }
        }
        if (all.length) {
            const diagonal = vec3.distance(min, max);
            viewDelta = Math.max(0.002, diagonal / 100);
        }
        setStatus(`Loaded ${triangleData.length} triangle sets and ${ellipsoidData.length} ellipsoids. Press ! for the Part 8 scene.`);
        window.requestAnimationFrame(renderFrame);
    } catch (error) {
        logError('Renderer initialization failed', error);
    }
}

window.addEventListener('load', main);
