export function log(msg: string, src = "express") {
  const t = new Date().toLocaleTimeString("en-US", { hour12: false });
  // eslint-disable-next-line no-console
  console.log(`${t} [${src}] ${msg}`);
} 