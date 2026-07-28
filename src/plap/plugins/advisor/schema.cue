package plap

#RegisteredSides: {
  advisor: 1024
}

#Config: {
  advisor?: #FieldConfig
  overlays: {
    model?: {
      [string]: {
        advisor!: #FieldConfig
        [string]: _
      }
    }
  }
}
