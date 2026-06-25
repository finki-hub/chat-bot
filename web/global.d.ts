// Side-effect global stylesheet imports (e.g. `import './globals.css'`). Next.js
// ships no ambient type for these, and `verbatimModuleSyntax` requires the module
// to be declared. The app uses only global CSS (no CSS Modules), so an opaque
// declaration is sufficient.
declare module '*.css';
