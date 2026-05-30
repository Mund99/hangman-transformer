import { type ReactNode } from 'react'

interface Props {
  headers: string[]
  rows: (ReactNode | string)[][]
}

export default function DocTable({ headers, rows }: Props) {
  return (
    <div className="doc-table-wrap">
      <table className="doc-table">
        <thead>
          <tr>{headers.map((h, i) => <th key={i}>{h}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri}>{row.map((cell, ci) => <td key={ci}>{cell}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
