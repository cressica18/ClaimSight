/// <reference types="vite/client" />

// CSS Module declarations — allows importing *.module.css files
declare module "*.module.css" {
  const classes: Record<string, string>;
  export default classes;
}

// Plain CSS imports (no type needed at runtime, just silence TS)
declare module "*.css" {
  const content: string;
  export default content;
}
