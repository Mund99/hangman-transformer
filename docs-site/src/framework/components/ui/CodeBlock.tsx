import { useState, useCallback, useEffect, useRef } from 'react'
import hljs from 'highlight.js/lib/core'
import typescript from 'highlight.js/lib/languages/typescript'
import javascript from 'highlight.js/lib/languages/javascript'
import python from 'highlight.js/lib/languages/python'
import bash from 'highlight.js/lib/languages/bash'
import json from 'highlight.js/lib/languages/json'
import yaml from 'highlight.js/lib/languages/yaml'
import xml from 'highlight.js/lib/languages/xml'
import css from 'highlight.js/lib/languages/css'
import 'highlight.js/styles/github-dark.css'

hljs.registerLanguage('typescript', typescript)
hljs.registerLanguage('tsx', typescript)
hljs.registerLanguage('javascript', javascript)
hljs.registerLanguage('jsx', javascript)
hljs.registerLanguage('python', python)
hljs.registerLanguage('bash', bash)
hljs.registerLanguage('shell', bash)
hljs.registerLanguage('json', json)
hljs.registerLanguage('yaml', yaml)
hljs.registerLanguage('yml', yaml)
hljs.registerLanguage('xml', xml)
hljs.registerLanguage('html', xml)
hljs.registerLanguage('css', css)

interface Props {
  language?: string
  /** Optional filename shown in the code block header (e.g. "src/app.tsx") */
  filename?: string
  children: string
}

export default function CodeBlock({ language = '', filename, children }: Props) {
  const [copied, setCopied] = useState(false)
  const codeRef = useRef<HTMLElement>(null)

  const copy = useCallback(() => {
    navigator.clipboard.writeText(children.trim()).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }, [children])

  useEffect(() => {
    if (!codeRef.current) return
    codeRef.current.textContent = children.trim()
    if (language && hljs.getLanguage(language)) {
      hljs.highlightElement(codeRef.current)
    }
  }, [children, language])

  return (
    <div className="code-block">
      <div className="code-header">
        <div className="code-header-left">
          <div className="code-dots">
            <span className="code-dot" />
            <span className="code-dot" />
            <span className="code-dot" />
          </div>
          {filename && <span className="code-filename">{filename}</span>}
        </div>
        <div className="code-header-right">
          {language && <span className="code-lang">{language}</span>}
          <button className={`code-copy ${copied ? 'ok' : ''}`} onClick={copy}>
            {copied ? '✓ copied' : 'copy'}
          </button>
        </div>
      </div>
      <div className="code-body">
        <pre><code ref={codeRef} className={language ? `language-${language}` : ''} /></pre>
      </div>
    </div>
  )
}
